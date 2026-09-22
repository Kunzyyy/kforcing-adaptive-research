"""Compare slim local tensor storage CRCs with historical ZIP directory metadata."""
import hashlib
import json
from pathlib import Path
import zipfile
from checkpoint_pickle_metadata import parse, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/public-history'


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    result = {}
    for model, filename in [('pflm','pflm_lm1b_k4.ckpt'),('ar','ar_best_lm1b.ckpt')]:
        directory = OUT/('historical_'+model)
        historical = read(directory/'metadata_summary.json')
        inspection = read(directory/'inspection.json')
        hist_entries = {x['name'].split('/data/')[-1]: x for x in inspection['entries'] if '/data/' in x['name']}
        current_path = Path('work/checkpoints')/filename
        with zipfile.ZipFile(current_path) as z:
            member = next(n for n in z.namelist() if n.endswith('/data.pkl'))
            raw = z.read(member)
            current = summarize(parse(raw))
            current_entries = {x.filename.split('/data/')[-1]: dict(bytes=x.file_size, crc32=x.CRC)
                               for x in z.infolist() if '/data/' in x.filename}
        assert set(current['state_dict']) == set(historical['state_dict'])
        comparisons, ema_index = [], 0
        for name, now in current['state_dict'].items():
            old = historical['state_dict'][name]
            assert (now['shape'],now['stride'],now['offset'],now['dtype']) == (old['shape'],old['stride'],old['offset'],old['dtype'])
            ne, oe = current_entries[now['storage_key']], hist_entries[old['storage_key']]
            item = dict(name=name, shape=now['shape'], bytes=ne['bytes'],
                current_crc32=ne['crc32'], historical_state_crc32=oe['crc32'],
                current_vs_historical_state_crc_equal=(ne['bytes'],ne['crc32'])==(oe['bytes'],oe['crc32']))
            if name != 'backbone.rotary_emb.inv_freq':
                ema = historical['ema']['shadow_params'][ema_index]
                assert (now['shape'],now['stride'],now['offset'],now['dtype']) == (ema['shape'],ema['stride'],ema['offset'],ema['dtype'])
                ee = hist_entries[ema['storage_key']]
                item.update(ema_positional_index=ema_index, historical_ema_crc32=ee['crc32'],
                    current_vs_positional_ema_crc_equal=(ne['bytes'],ne['crc32'])==(ee['bytes'],ee['crc32']))
                ema_index += 1
            comparisons.append(item)
        assert ema_index == len(historical['ema']['shadow_params'])
        result[model] = dict(current_checkpoint_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
            current_pickle_sha256=hashlib.sha256(raw).hexdigest(), state_tensors=len(comparisons),
            historical_state_crc_matches=sum(x['current_vs_historical_state_crc_equal'] for x in comparisons),
            ema_parameters=ema_index, positional_ema_crc_matches=sum(x.get('current_vs_positional_ema_crc_equal',False) for x in comparisons),
            comparisons=comparisons)
    result['limitations'] = ('CRC32 equality is a screening result, not cryptographic or bytewise equality. '
        'EMA is an unnamed list; its mapping is a shape-checked positional hypothesis excluding rotary buffer, '
        'not verified training parameter registration order. No weight substitution or quality evaluation was performed.')
    path = OUT/'storage_comparison.json'
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='comparisons'} if isinstance(v,dict) else v for k,v in result.items()},indent=2))


if __name__ == '__main__': main()
