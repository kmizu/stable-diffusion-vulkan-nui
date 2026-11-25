package sdfx.vulkan.tools;

import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.stream.IntStream;

/** Torchvision RAFT inference. The learned encoders and recurrent update run on the selected ncnn device. */
public final class RaftFlow {
    private final int fw, fh, w, h, n, features, hidden, context, radius, corr;
    private final boolean small;
    private final String device;
    private final Path bridge, model;
    private final Runnable check;

    public RaftFlow(int width, int height, boolean small, String device, Path bridge, Path model, Runnable check) {
        fw = Math.max(128, width / 16 * 16);
        fh = Math.max(128, height / 16 * 16);
        w = fw / 8; h = fh / 8; n = w * h;
        this.small = small; this.device = device; this.bridge = bridge; this.model = model; this.check = check;
        features = small ? 128 : 256;
        hidden = small ? 96 : 128;
        context = small ? 64 : 128;
        radius = small ? 3 : 4;
        corr = 4 * (2 * radius + 1) * (2 * radius + 1);
    }

    private float[] run(String part, float[] data, int width, int height, int outputs) throws Exception {
        return NcnnTensor.run(bridge, device, model, part, data, width, height, outputs, check);
    }

    private static float lerp(float a, float b, double t) { return (float)(a+(b-a)*t); }
    private static int clamp(int i, int len) { return Math.min(len-1,Math.max(i,0)); }

    public float[] encode(BufferedImage im) throws Exception {
        int iw=im.getWidth(), ih=im.getHeight(), stride=fw*fh;
        int[] pixels=im.getRGB(0,0,iw,ih,null,0,iw);
        float[] rgb=new float[stride*3];
        for (int y=0;y<fh;y++) for(int x=0;x<fw;x++) {
            double sx=(x+.5)*iw/fw-.5, sy=(y+.5)*ih/fh-.5;
            int ix=(int)Math.floor(sx), iy=(int)Math.floor(sy);
            int a=pixels[clamp(iy,ih)*iw+clamp(ix,iw)], b=pixels[clamp(iy,ih)*iw+clamp(ix+1,iw)];
            int c=pixels[clamp(iy+1,ih)*iw+clamp(ix,iw)], d=pixels[clamp(iy+1,ih)*iw+clamp(ix+1,iw)];
            for (int channel=0;channel<3;channel++) {
                int shift=16-8*channel;
                float v=lerp(lerp((a>>shift)&255,(b>>shift)&255,sx-ix),lerp((c>>shift)&255,(d>>shift)&255,sx-ix),sy-iy);
                rgb[channel*stride+y*fw+x]=Math.round(v)/127.5f-1;
            }
        }
        return run("encoder",rgb,fw,fh,n*(features+hidden+context));
    }

    private static float at(float[] array,int off,int w,int h,int x,int y) {
        return x<0||x>=w||y<0||y>=h?0:array[off+y*w+x];
    }
    private static float bilinear(float[] input,int off,int w,int h,double x,double y) {
        int xx=(int)Math.floor(x), yy=(int)Math.floor(y);
        return lerp(lerp(at(input,off,w,h,xx,yy),at(input,off,w,h,xx+1,yy),x-xx),
                    lerp(at(input,off,w,h,xx,yy+1),at(input,off,w,h,xx+1,yy+1),x-xx),y-yy);
    }

    private final class Correlation {
        final float[][] volumes = new float[4][];
        final float[] source, target;
        final float[][] targetLevels = new float[4][];
        final float scale = (float)(1/Math.sqrt(features));
        final int[] widths = { w,w/2,w/4,w/8 };
        final int[] heights = { h,h/2,h/4,h/8 };
        Correlation(float[] a,float[] b) {
            source = new float[n*features]; target = new float[n*features];
            for (int i=0;i<n;i++) for (int ch=0;ch<features;ch++) {
                source[i*features+ch]=a[ch*n+i]; target[i*features+ch]=b[ch*n+i];
            }
            if ((long)n*n <= 26000000) {
                volumes[0]=new float[n*n];
                IntStream.range(0,n).parallel().forEach(i -> {
                    check.run();
                    for(int j=0;j<n;j++) volumes[0][i*n+j]=dot(i,j);
                });
                for(int level=1;level<4;level++) {
                    int ww=widths[level], hh=heights[level], prevW=widths[level-1], prevH=heights[level-1];
                    float[] prev=volumes[level-1], next=new float[n*ww*hh];
                    for(int i=0;i<n;i++) for(int y=0;y<hh;y++) for(int x=0;x<ww;x++) {
                        int p=(i*prevH+2*y)*prevW+2*x;
                        next[(i*hh+y)*ww+x]=(prev[p]+prev[p+1]+prev[p+prevW]+prev[p+prevW+1])*.25f;
                    }
                    volumes[level]=next;
                }
            } else {
                // Pool the target features before taking dot products. Dot products are linear,
                // so this is equivalent to pooling the dense all-pairs correlation tensor.
                targetLevels[0]=target;
                for (int level=1;level<4;level++) {
                    int ww=widths[level], hh=heights[level], prevW=widths[level-1];
                    float[] prev=targetLevels[level-1], next=new float[ww*hh*features];
                    for(int y=0;y<hh;y++) for(int x=0;x<ww;x++) {
                        int base=(2*y*prevW+2*x)*features;
                        for(int c=0;c<features;c++)
                            next[(y*ww+x)*features+c]=.25f*(prev[base+c]+prev[base+c+features]
                                +prev[base+c+prevW*features]+prev[base+c+(prevW+1)*features]);
                    }
                    targetLevels[level]=next;
                }
            }
        }
        float dot(int i,int j) {
            return dot(i,j,target);
        }
        float dot(int i,int j,float[] values) {
            float v=0;
            for(int ch=0;ch<features;ch++) v+=source[i*features+ch]*values[j*features+ch];
            return v*scale;
        }
        void sample(float[] state,int offset,float[] flow) {
            IntStream.range(0,n).parallel().forEach(i -> {
                check.run();
                double cx=i%w+flow[i], cy=i/w+flow[n+i];
                int channel=0;
                for(int level=0;level<4;level++) {
                    int ww=widths[level],hh=heights[level];
                    float[] plane=volumes[level];
                    int base=i*ww*hh;
                    double xx=cx, yy=cy;
                    if(plane==null) {
                        int startX=(int)Math.floor(cx)-radius, startY=(int)Math.floor(cy)-radius;
                        int side=2*radius+2;
                        plane=new float[side*side];
                        for(int y=0;y<side;y++) for(int x=0;x<side;x++) {
                            int px=startX+x, py=startY+y;
                            if (px>=0 && px<ww && py>=0 && py<hh)
                                plane[y*side+x]=dot(i,py*ww+px,targetLevels[level]);
                        }
                        xx-=startX; yy-=startY;
                        ww=side; hh=side;
                        base=0;
                    }
                    for(int dx=-radius;dx<=radius;dx++) for(int dy=-radius;dy<=radius;dy++)
                        state[(offset+channel++)*n+i]=bilinear(plane,base,ww,hh,xx+dx,yy+dy);
                    cx*=.5; cy*=.5;
                }
            });
        }
    }

    public float[] predict(float[] previous,float[] current,int iters,int width,int height) throws Exception {
        Correlation pyramid=new Correlation(previous,current);
        int hc=hidden+context;
        float[] state=new float[(hc+corr+2)*n], flow=new float[2*n], update=null;
        for(int i=0;i<hidden*n;i++) state[i]=(float)Math.tanh(previous[features*n+i]);
        for(int i=hidden*n;i<hc*n;i++) state[i]=Math.max(0,previous[features*n+i]);
        for(int step=0;step<iters;step++) {
            check.run();
            pyramid.sample(state,hc,flow);
            System.arraycopy(flow,0,state,(hc+corr)*n,2*n);
            update=run("update",state,w,h,(hidden+2+(small?0:576))*n);
            System.arraycopy(update,0,state,0,hidden*n);
            for(int i=0;i<flow.length;i++) flow[i]+=update[hidden*n+i];
        }
        float[] high=new float[2*fw*fh];
        for(int y=0;y<fh;y++) for(int x=0;x<fw;x++) {
            int p=(y/8)*w+x/8, sub=(y%8)*8+x%8;
            if(small) {
                for(int c=0;c<2;c++) high[c*fw*fh+y*fw+x]=8*bilinear(flow,c*n,w,h,(double)x*(w-1)/(fw-1),(double)y*(h-1)/(fh-1));
            } else {
                float max=Float.NEGATIVE_INFINITY;
                for(int j=0;j<9;j++) max=Math.max(max,update[(hidden+2+j*64+sub)*n+p]);
                double sum=0, sx=0, sy=0;
                for(int j=0;j<9;j++) {
                    double weight=Math.exp(update[(hidden+2+j*64+sub)*n+p]-max);
                    sum+=weight;
                    sx+=weight*at(flow,0,w,h,x/8+j%3-1,y/8+j/3-1);
                    sy+=weight*at(flow,n,w,h,x/8+j%3-1,y/8+j/3-1);
                }
                high[y*fw+x]=(float)(8*sx/sum); high[fw*fh+y*fw+x]=(float)(8*sy/sum);
            }
        }
        float[] result=new float[2*width*height];
        for(int c=0;c<2;c++) for(int y=0;y<height;y++) for(int x=0;x<width;x++) {
            double xx=Math.max(0,Math.min(fw-1,(x+.5)*fw/width-.5)), yy=Math.max(0,Math.min(fh-1,(y+.5)*fh/height-.5));
            result[c*width*height+y*width+x]=bilinear(high,c*fw*fh,fw,fh,xx,yy)*(c==0?(float)width/fw:(float)height/fh);
        }
        return result;
    }
}
