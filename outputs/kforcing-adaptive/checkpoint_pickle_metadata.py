"""Non-executing, deliberately narrow pickle opcode interpreter for metadata.

GLOBAL/REDUCE/NEWOBJ/BUILD are symbolic records, never imports or function calls.
Unknown opcodes fail closed. Tensor storage bytes are not read by this module.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile


@dataclass
class Global:
    name: str


@dataclass
class Symbol:
    kind: str
    args: object = None
    state: object = None
    items: dict = field(default_factory=dict)


def parse(raw):
    stack, memo, mark = [], {}, object()
    def pop_mark():
        pos = len(stack)-1
        while stack[pos] is not mark: pos -= 1
        result = stack[pos+1:]
        del stack[pos:]
        return result
    def target(obj): return obj.items if isinstance(obj, Symbol) else obj
    for op, arg, pos in pickletools.genops(raw):
        code = op.name
        if code in {'PROTO', 'FRAME'}: pass
        elif code == 'MARK': stack.append(mark)
        elif code in {'BININT','BININT1','BININT2','LONG1','LONG4','BINFLOAT','BINUNICODE','SHORT_BINUNICODE','UNICODE','BINBYTES','SHORT_BINBYTES'}: stack.append(arg)
        elif code == 'NONE': stack.append(None)
        elif code == 'NEWTRUE': stack.append(True)
        elif code == 'NEWFALSE': stack.append(False)
        elif code == 'EMPTY_DICT': stack.append({})
        elif code == 'EMPTY_LIST': stack.append([])
        elif code == 'EMPTY_TUPLE': stack.append(())
        elif code == 'TUPLE': stack.append(tuple(pop_mark()))
        elif code in {'TUPLE1','TUPLE2','TUPLE3'}:
            n = int(code[-1]); value = tuple(stack[-n:]); del stack[-n:]; stack.append(value)
        elif code in {'BINPUT', 'LONG_BINPUT'}: memo[arg] = stack[-1]
        elif code == 'MEMOIZE': memo[len(memo)] = stack[-1]
        elif code in {'BINGET', 'LONG_BINGET'}: stack.append(memo[arg])
        elif code == 'APPEND':
            value = stack.pop(); stack[-1].append(value)
        elif code == 'APPENDS':
            values = pop_mark(); stack[-1].extend(values)
        elif code == 'SETITEM':
            value, key = stack.pop(), stack.pop(); target(stack[-1])[key] = value
        elif code == 'SETITEMS':
            values = pop_mark()
            if len(values) % 2: raise ValueError('odd SETITEMS')
            target(stack[-1]).update(zip(values[::2], values[1::2]))
        elif code == 'GLOBAL': stack.append(Global(arg))
        elif code == 'STACK_GLOBAL':
            name, module = stack.pop(), stack.pop(); stack.append(Global(module+' '+name))
        elif code in {'REDUCE', 'NEWOBJ'}:
            values, constructor = stack.pop(), stack.pop()
            if not isinstance(constructor, Global): raise ValueError('non-global symbolic constructor')
            stack.append(Symbol(constructor.name, values))
        elif code == 'BUILD':
            value = stack.pop()
            if not isinstance(stack[-1], Symbol): raise ValueError('BUILD needs symbolic object')
            stack[-1].state = value
        elif code == 'BINPERSID': stack.append(Symbol('persistent_storage', stack.pop()))
        elif code == 'STOP':
            if len(stack) != 1: raise ValueError('invalid final stack')
            return stack[0]
        else: raise ValueError(f'Unsupported opcode {code} at {pos}')
    raise ValueError('missing STOP')


def plain(obj, active=None):
    active = set() if active is None else active
    if obj is None or isinstance(obj, (bool, int, float, str)): return obj
    if id(obj) in active: return '<cycle>'
    active = active | {id(obj)}
    if isinstance(obj, Global): return {'symbolic_global': obj.name}
    if isinstance(obj, (list, tuple)): return [plain(x, active) for x in obj]
    if isinstance(obj, dict): return {str(k): plain(v, active) for k, v in obj.items() if k != '_parent'}
    if isinstance(obj, Symbol):
        if obj.kind.startswith('omegaconf.') and isinstance(obj.state, dict):
            if '_content' in obj.state: return plain(obj.state['_content'], active)
            if '_val' in obj.state: return plain(obj.state['_val'], active)
        if obj.kind in {'collections OrderedDict', 'collections defaultdict'}: return plain(obj.items, active)
        return dict(symbolic_type=obj.kind, args=plain(obj.args, active), state=plain(obj.state, active), items=plain(obj.items, active))
    raise TypeError(type(obj))


def tensor(obj):
    if not isinstance(obj, Symbol) or obj.kind != 'torch._utils _rebuild_tensor_v2':
        raise ValueError('expected tensor descriptor')
    storage, offset, shape, stride, *_ = obj.args
    if not isinstance(storage, Symbol) or storage.kind != 'persistent_storage': raise ValueError('expected storage')
    tag, dtype, key, location, elements = storage.args
    assert tag == 'storage'
    return dict(storage_key=key, dtype=dtype.name, storage_elements=elements,
                offset=offset, shape=list(shape), stride=list(stride), original_device=location)


def items(obj): return obj.items if isinstance(obj, Symbol) else obj


def summarize(metadata):
    states = {k: tensor(v) for k, v in items(metadata['state_dict']).items()}
    ema = metadata.get('ema')
    return dict(top_level_keys=list(metadata), epoch=metadata.get('epoch'), global_step=metadata.get('global_step'),
        lightning_version=metadata.get('pytorch-lightning_version'), state_dict=states,
        ema=None if ema is None else dict(decay=ema['decay'], num_updates=ema.get('num_updates'),
            shadow_params=[tensor(x) for x in ema['shadow_params']]),
        hyper_parameters=plain({k:v for k,v in metadata.get('hyper_parameters',{}).items() if k != 'tokenizer'}),
        lr_schedulers=plain(metadata.get('lr_schedulers')), callbacks=plain(metadata.get('callbacks')),
        note='Symbolic syntax only. No GLOBAL imports, REDUCE calls, or pickle object execution. OmegaConf interpolation strings are unresolved.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickle_path', type=Path)
    args = ap.parse_args()
    raw = args.pickle_path.read_bytes()
    result = summarize(parse(raw))
    result['pickle_sha256'] = hashlib.sha256(raw).hexdigest()
    out = args.pickle_path.with_name('metadata_summary.json')
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['epoch','global_step','lightning_version','top_level_keys']}, indent=2))
    print('state tensors',len(result['state_dict']),'EMA tensors',len(result['ema']['shadow_params']) if result['ema'] else 0)


if __name__ == '__main__': main()
