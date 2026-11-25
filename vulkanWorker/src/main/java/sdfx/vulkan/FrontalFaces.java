package sdfx.vulkan;

import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/** Scans an image pyramid with OpenCV's upright frontal-face Haar cascade. */
public final class FrontalFaces {
    private record Feature(int x, int y, int w, int h, float weight) {}
    private record Stump(Feature[] feature, float threshold, float less, float greater) {}
    private record Stage(float threshold, Stump[] stumps) {}
    private static final int WINDOW = 24;
    private static final List<Stage> STAGES = load();

    private static List<Element> children(Element parent) {
        var result = new ArrayList<Element>();
        for (Node node = parent.getFirstChild(); node != null; node = node.getNextSibling())
            if (node instanceof Element el && el.getTagName().equals("_")) result.add(el);
        return result;
    }
    private static Element child(Element e, String key) {
        return (Element)e.getElementsByTagName(key).item(0);
    }
    private static String[] fields(Element e) { return e.getTextContent().strip().split("\\s+"); }
    private static List<Stage> load() {
        try (var xml = FrontalFaces.class.getResourceAsStream("/sdfx/vulkan/faces/haarcascade_frontalface_default.xml")) {
            var parser = DocumentBuilderFactory.newInstance();
            parser.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            var cascade = child(parser.newDocumentBuilder().parse(xml).getDocumentElement(), "cascade");
            var features = new ArrayList<Feature[]>();
            for (var feature : children(child(cascade, "features"))) {
                var rects = new ArrayList<Feature>();
                for (var el : children(child(feature, "rects"))) {
                    var r = fields(el);
                    rects.add(new Feature(Integer.parseInt(r[0]), Integer.parseInt(r[1]), Integer.parseInt(r[2]),
                        Integer.parseInt(r[3]), Float.parseFloat(r[4])));
                }
                features.add(rects.toArray(Feature[]::new));
            }
            var stages = new ArrayList<Stage>();
            for (var stage : children(child(cascade,"stages"))) {
                var classifiers = new ArrayList<Stump>();
                for (var weak : children(child(stage,"weakClassifiers"))) {
                    var node = fields(child(weak,"internalNodes"));
                    var leaves = fields(child(weak,"leafValues"));
                    classifiers.add(new Stump(features.get(Integer.parseInt(node[2])), Float.parseFloat(node[3]),
                        Float.parseFloat(leaves[0]),Float.parseFloat(leaves[1])));
                }
                stages.add(new Stage(Float.parseFloat(child(stage,"stageThreshold").getTextContent().strip())-1e-5f,
                    classifiers.toArray(Stump[]::new)));
            }
            return List.copyOf(stages);
        } catch (Exception e) { throw new ExceptionInInitializerError(e); }
    }
    private static long area(long[] integral, int stride, int x, int y, int w, int h) {
        int start = y * stride + x;
        return integral[start] - integral[start+w] - integral[start+h*stride] + integral[start+h*stride+w];
    }
    private static boolean face(long[] sums, long[] squares, int stride, int x, int y) {
        long sum = area(sums,stride,x+1,y+1,22,22);
        double norm = Math.sqrt(484.0*area(squares,stride,x+1,y+1,22,22)-sum*sum);
        if (!(norm > 4840)) return false;
        for (var stage : STAGES) {
            float total = 0;
            for (var c : stage.stumps) {
                double value = 0;
                for (var r : c.feature) value += area(sums,stride,x+r.x,y+r.y,r.w,r.h)*r.weight;
                total += value < norm*c.threshold ? c.less : c.greater;
            }
            if (total < stage.threshold) return false;
        }
        return true;
    }
    public static List<Rectangle> detect(BufferedImage src, Runnable check) {
        var hits = new ArrayList<Rectangle>();
        int mw = src.getWidth(), mh = src.getHeight(), minimum = Math.max(24, Math.min(mw,mh)/20);
        if ((long)mw*mh > 32_000_000) throw new IllegalArgumentException("Face detector pixel limit exceeded");
        for (double scale = 1; WINDOW*scale <= Math.min(mw,mh); scale *= 1.1) {
            int size = (int)Math.round(scale*WINDOW);
            if (size < minimum) continue;
            int w=(int)Math.round(mw/scale), h=(int)Math.round(mh/scale), stride=w+1;
            var small = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
            Graphics2D g = small.createGraphics();
            try {
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(src,0,0,w,h,null);
            } finally { g.dispose(); }
            long[] sums = new long[stride*(h+1)], squares=new long[sums.length];
            for (int y=0;y<h;y++) {
                check.run();
                long sum=0,sq=0;
                for (int x=0;x<w;x++) {
                    int rgb=small.getRGB(x,y), v=(77*((rgb>>16)&255)+150*((rgb>>8)&255)+29*(rgb&255)+128)>>8;
                    sum+=v; sq+=v*v;
                    int i=(y+1)*stride+x+1; sums[i]=sums[i-stride]+sum; squares[i]=squares[i-stride]+sq;
                }
            }
            int step=scale>2 ? 1 : 2;
            for (int y=0;y<=h-WINDOW;y+=step) {
                check.run();
                for (int x=0;x<=w-WINDOW;x+=step) if (face(sums,squares,stride,x,y)) {
                    if (hits.size()>=8192) throw new IllegalArgumentException("Too many candidate faces");
                    hits.add(new Rectangle((int)Math.round(scale*x),(int)Math.round(scale*y),size,size));
                }
            }
        }
        return group(hits,check);
    }
    private static int root(int[] ids,int i) {
        while (ids[i]!=i) { ids[i]=ids[ids[i]]; i=ids[i]; }
        return i;
    }
    private static List<Rectangle> group(List<Rectangle> hits,Runnable check) {
        int n=hits.size(); int[] ids=new int[n]; Arrays.setAll(ids,i -> i);
        for (int i=0;i<n;i++) {
            check.run();
            var a=hits.get(i);
            for(int j=0;j<i;j++) {
                var b=hits.get(j); double d=0.2*Math.min(a.width,b.width);
                if(Math.abs(a.x-b.x)<=d && Math.abs(a.y-b.y)<=d && Math.abs(a.x+a.width-b.x-b.width)<=d &&
                    Math.abs(a.y+a.height-b.y-b.height)<=d) ids[root(ids,i)]=root(ids,j);
            }
        }
        long[][] totals=new long[n][4];
        for(int i=0;i<n;i++) {
            var r=hits.get(i);var t=totals[root(ids,i)];
            t[0]+=r.x;t[1]+=r.y;t[2]+=r.width;t[3]++;
        }
        var result=new ArrayList<Rectangle>();
        for(var t:totals) if(t[3]>5) {
            int side=(int)Math.round((double)t[2]/t[3]);
            result.add(new Rectangle((int)Math.round((double)t[0]/t[3]),(int)Math.round((double)t[1]/t[3]),side,side));
        }
        return result;
    }
}
