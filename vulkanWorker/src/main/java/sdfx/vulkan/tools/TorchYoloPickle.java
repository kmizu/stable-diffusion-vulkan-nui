package sdfx.vulkan.tools;

import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Reads the subset of pickle protocol 2 used by Ultralytics state dictionaries. Globals are inert names. */
public final class TorchYoloPickle {
    public record Tensor(String storage, String dtype, int offset, List<Integer> shape, List<Integer> stride) {}
    private record Global(String name) {}
    private record Storage(String key, String dtype) {}
    public static final class Module {
        public final String type;
        public Map<Object,Object> state = Map.of();
        Module(String type) { this.type = type; }
        public Map<Object,Object> children() { return map(state.get("_modules")); }
    }
    private final ArrayList<Object> stack = new ArrayList<>();
    private final Map<Integer,Object> memo = new HashMap<>();
    private final Object mark = new Object();
    private final ByteBuffer data;
    private TorchYoloPickle(byte[] bytes) { data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN); }
    /** Interprets the tensor descriptors used by torch.save without constructing Python objects. */
    public static Object tensorPayload(byte[] data) {
        return new TorchYoloPickle(data).read();
    }
    public static Module model(byte[] data) {
        Map<Object,Object> root = map(new TorchYoloPickle(data).read());
        Object m = root.get("ema") instanceof Module ? root.get("ema") : root.get("model");
        if (!(m instanceof Module net) || !net.type.endsWith(".DetectionModel"))
            throw new IllegalArgumentException("Expected a YOLOv8 DetectionModel checkpoint");
        return net;
    }
    private static IllegalArgumentException invalid() { return new IllegalArgumentException("Unsupported YOLO checkpoint data"); }
    @SuppressWarnings("unchecked")
    public static Map<Object,Object> map(Object v) {
        if (!(v instanceof Map)) throw invalid();
        return (Map<Object,Object>)v;
    }
    @SuppressWarnings("unchecked")
    private static List<Object> list(Object v) {
        if (!(v instanceof List)) throw invalid();
        return (List<Object>)v;
    }
    private static List<Integer> dims(Object v) {
        return list(v).stream().map(x -> ((Number)x).intValue()).toList();
    }
    private Object pop() { return stack.remove(stack.size()-1); }
    private Object peek() { return stack.get(stack.size()-1); }
    private void push(Object value) { stack.add(value); }
    private int octet() { return Byte.toUnsignedInt(data.get()); }
    private String name() {
        int start = data.position();
        while (data.get() != 10) if (data.position()-start > 512) throw invalid();
        return new String(data.array(), start, data.position()-start-1, StandardCharsets.US_ASCII);
    }
    private List<Object> tail(int start) {
        var slice = stack.subList(start,stack.size());
        var result = new ArrayList<>(slice);
        slice.clear();
        return result;
    }
    private List<Object> marked() {
        int index = stack.lastIndexOf(mark);
        if (index < 0) throw invalid();
        var result = tail(index+1); pop();
        return result;
    }
    private Object reduce(Global fn, List<Object> args) {
        return switch (fn.name()) {
            case "collections.OrderedDict" -> new LinkedHashMap<>();
            case "__builtin__.set", "torch.Size", "torch.device" -> args;
            case "torch._utils._rebuild_parameter" -> args.get(0);
            case "torch._utils._rebuild_tensor_v2" -> {
                Storage s = (Storage)args.get(0);
                yield new Tensor(s.key(),s.dtype(),((Number)args.get(1)).intValue(),dims(args.get(2)),dims(args.get(3)));
            }
            default -> throw invalid();
        };
    }
    private Object read() {
        while (data.hasRemaining()) {
            if (stack.size()>200000 || memo.size()>200000) throw invalid();
            int op = octet();
            switch(op) {
                case 128 -> { if (octet()!=2) throw invalid(); }
                case '.' -> { if(stack.size()!=1) throw invalid(); return pop(); }
                case '(' -> push(mark);
                case '}' -> push(new LinkedHashMap<>());
                case ']',')' -> push(new ArrayList<>());
                case 'N' -> push(null);
                case 136,137 -> push(op==136);
                case 'K' -> push(octet());
                case 'M' -> push(Short.toUnsignedInt(data.getShort()));
                case 'J' -> push(data.getInt());
                case 'G' -> push(Double.longBitsToDouble(Long.reverseBytes(data.getLong())));
                case 'X' -> {
                    int n = data.getInt();
                    if (n<0 || n>1048576 || n>data.remaining()) throw invalid();
                    byte[] b=new byte[n];data.get(b);push(new String(b,StandardCharsets.UTF_8));
                }
                case 'c' -> push(new Global(name()+"."+name()));
                case 'q','r' -> memo.put(op=='q' ? octet() : data.getInt(),peek());
                case 'h','j' -> {
                    int id=op=='h'?octet():data.getInt();
                    if(!memo.containsKey(id)) throw invalid(); push(memo.get(id));
                }
                case 133,134,135 -> push(tail(stack.size()-(op-132)));
                case 't' -> push(marked());
                case 'a' -> { Object v=pop();list(peek()).add(v); }
                case 'e' -> { var values=marked();list(peek()).addAll(values); }
                case 's' -> { Object v=pop(),k=pop();map(peek()).put(k,v); }
                case 'u' -> {
                    var values=marked();var table=map(peek());
                    for(int i=0;i<values.size();i+=2) table.put(values.get(i),values.get(i+1));
                }
                case 'Q' -> {
                    var pid=list(pop());
                    if (!"storage".equals(pid.get(0))) throw invalid();
                    String key=(String)pid.get(2), type=((Global)pid.get(1)).name();
                    if (!key.matches("[0-9]+")) throw invalid();
                    push(new Storage(key,type));
                }
                case 'R' -> { var args=list(pop());push(reduce((Global)pop(),args)); }
                case 129 -> {
                    var args=list(pop());var type=((Global)pop()).name();
                    if (!args.isEmpty() || !(type.startsWith("torch.nn.") || type.startsWith("ultralytics."))) throw invalid();
                    push(new Module(type));
                }
                case 'b' -> { var state=map(pop()); ((Module)peek()).state=state; }
                default -> throw invalid();
            }
        }
        throw invalid();
    }
}
