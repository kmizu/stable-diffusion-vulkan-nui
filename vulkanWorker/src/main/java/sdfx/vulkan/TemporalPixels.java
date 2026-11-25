package sdfx.vulkan;

import java.awt.image.BufferedImage;
import java.util.Arrays;

/** RGB frame arithmetic used by Temporal Lab. The mask is 255 at stable pixels. */
public final class TemporalPixels {
    private TemporalPixels() {}
    public static int[] rgb(BufferedImage image) {
        return image.getRGB(0,0,image.getWidth(),image.getHeight(),null,0,image.getWidth());
    }
    public static BufferedImage image(int[] pixels,int w,int h) {
        var out = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        out.setRGB(0,0,w,h,pixels,0,w);
        return out;
    }
    private static int channel(int rgb,int shift) { return (rgb>>>shift)&255; }
    private static int gray(int rgb) {
        return (4899*channel(rgb,16) + 9617*channel(rgb,8) + 1868*channel(rgb,0) + 8192) >> 14;
    }
    public static double meanGray(int[] rgb) {
        long sum=0;
        for (int value : rgb) sum+=gray(value);
        return sum/(double)rgb.length;
    }
    public static int[] overlay(int[] style,int[] current,double strength,boolean brighten) {
        var result = new int[current.length];
        for (int i=0;i<result.length;i++) {
            int rgb=0;
            for (int shift=0;shift<=16;shift+=8) {
                double a=channel(style[i],shift)/255.0,b=channel(current[i],shift)/255.0;
                double comp=a<=0.5 ? 2*a*b : 1-2*(1-a)*(1-b);
                int value=(int)Math.clamp((b*(1-strength)+comp*strength)*255,0,255);
                rgb |= Math.min(255,value+(brighten?10:0))<<shift;
            }
            result[i]=rgb;
        }
        return result;
    }
    public static int[] mix(int[] previous,int[] current,int[] mask,double retention) {
        var out=new int[current.length];
        for (int i=0;i<out.length;i++) {
            double a=mask==null?retention:(float)(mask[i]/255f*(float)retention);
            for (int shift=0;shift<=16;shift+=8)
                out[i] |= ((int)Math.rint(channel(previous[i],shift)*a+channel(current[i],shift)*(1-a)))<<shift;
        }
        return out;
    }
    private static int reflect(int i,int n) {
        if(n==1)return 0;
        int p=2*n-2, wrapped=Math.floorMod(i,p);
        return wrapped<n?wrapped:p-wrapped;
    }
    public static int[] gaussian(int[] src,int w,int h,int kernel,boolean rgb,Runnable check) {
        if(kernel<=1) return src;
        int radius=kernel/2;
        double[] weights=switch(kernel) {
            case 3 -> new double[]{1,2,1};
            case 5 -> new double[]{1,4,6,4,1};
            case 7 -> new double[]{2,7,14,18,14,7,2};
            case 9 -> new double[]{4,13,30,51,60,51,30,13,4};
            default -> new double[kernel];
        };
        if(kernel>9) {
            double sigma=0.3*(radius-1)+0.8;
            for(int i=0;i<kernel;i++)weights[i]=Math.exp(-.5*(i-radius)*(i-radius)/(sigma*sigma));
        }
        double sum=Arrays.stream(weights).sum();
        for(int i=0;i<kernel;i++)weights[i]/=sum;
        int[] out=new int[src.length];
        float[] scratch=new float[src.length];
        for(int shift=0;shift<(rgb?24:8);shift+=8) {
            for(int y=0;y<h;y++) {
                check.run();
                for(int x=0;x<w;x++) {
                    double value=0;
                    for(int j=-radius;j<=radius;j++)value+=channel(src[y*w+reflect(x+j,w)],shift)*weights[j+radius];
                    scratch[y*w+x]=(float)value;
                }
            }
            for(int y=0;y<h;y++) {
                check.run();
                for(int x=0;x<w;x++) {
                    double value=0;
                    for(int j=-radius;j<=radius;j++)value+=scratch[reflect(y+j,h)*w+x]*weights[j+radius];
                    out[y*w+x]|=((int)Math.round(value))<<shift;
                }
            }
        }
        return out;
    }
    public static int[] bilateral(int[] src,int w,int h,int diameter,double sigma,Runnable check) {
        int radius=Math.max(diameter/2,1);
        double[] color=new double[766];
        for(int i=0;i<color.length;i++)color[i]=Math.exp(-.5*i*i/(sigma*sigma));
        int capacity=(2*radius+1)*(2*radius+1), count=0;
        int[] dx=new int[capacity],dy=new int[capacity]; double[] space=new double[capacity];
        for(int y=-radius;y<=radius;y++)for(int x=-radius;x<=radius;x++)
            if(x*x+y*y<=radius*radius) {
                dx[count]=x; dy[count]=y; space[count++]=Math.exp(-.5*(x*x+y*y)/(sigma*sigma));
            }
        int[] out=new int[src.length];
        for(int y=0;y<h;y++) {
            check.run();
            for(int x=0;x<w;x++) {
                int center=src[y*w+x]; double r=0,g=0,b=0,sum=0;
                for(int k=0;k<count;k++) {
                    int n=src[reflect(y+dy[k],h)*w+reflect(x+dx[k],w)];
                    int distance=Math.abs(channel(n,16)-channel(center,16))+Math.abs(channel(n,8)-channel(center,8))+Math.abs(channel(n,0)-channel(center,0));
                    double weight=color[distance]*space[k]; sum+=weight;
                    r+=channel(n,16)*weight;g+=channel(n,8)*weight;b+=channel(n,0)*weight;
                }
                out[y*w+x]=((int)Math.rint(r/sum)<<16)|((int)Math.rint(g/sum)<<8)|(int)Math.rint(b/sum);
            }
        }
        return out;
    }
    public static int[] mask(int[] previous,int[] current,int w,int h,int denoise,double tolerance,int expand,int smooth,Runnable check) {
        if(denoise>0) {
            previous=bilateral(previous,w,h,denoise,31,check);
            current=bilateral(current,w,h,denoise,31,check);
        }
        int[] stable=new int[current.length];
        for(int i=0;i<stable.length;i++)stable[i]=Math.abs(gray(previous[i])-gray(current[i]))<=tolerance?255:0;
        if(expand>1) {
            // A summed-area table makes dilation of changing pixels independent of the kernel area.
            int stride=w+1; int[] integral=new int[(h+1)*stride];
            for(int y=0;y<h;y++) {
                int row=0;
                for(int x=0;x<w;x++) {
                    if(stable[y*w+x]==0)row++;
                    integral[(y+1)*stride+x+1]=row+integral[y*stride+x+1];
                }
            }
            for(int y=0;y<h;y++) {
                check.run();
                int top=Math.max(0,y-expand/2),bottom=Math.min(h,y-expand/2+expand);
                for(int x=0;x<w;x++) {
                    int l=Math.max(0,x-expand/2),r=Math.min(w,x-expand/2+expand);
                    int changes=integral[bottom*stride+r]-integral[top*stride+r]-integral[bottom*stride+l]+integral[top*stride+l];
                    stable[y*w+x]=changes==0?255:0;
                }
            }
        }
        return gaussian(stable,w,h,smooth,false,check);
    }
}
