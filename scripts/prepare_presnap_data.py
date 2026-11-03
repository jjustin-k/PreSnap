"""Build the pre-snap coverage dataset. Run: uv run --with pandas python scripts/prepare_presnap_data.py.

If raw files are missing, download the pinned regional release with --download.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

REVISION = '85da22eeff2f1d5be106faa9dfe06a1205f2defd'
SOURCE = 'https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data'
CONTEXT = ['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'possessionTeam',
           'defensiveTeam', 'yardlineSide', 'yardlineNumber', 'gameClock',
           'preSnapHomeScore', 'preSnapVisitorScore', 'absoluteYardlineNumber',
           'offenseFormation', 'personnelO', 'personnelD', 'defendersInBox']
TRACKING = ['gameId', 'playId', 'frameId', 'nflId', 'team', 'playDirection',
            'x', 'y', 's', 'a', 'o', 'dir']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    raw = Path('data/regional-2021')
    out = Path('data/presnap')
    manifest = json.loads((out / 'source_manifest.json').read_text())
    needed = [f for f in manifest['files'] if f['path'] == 'plays.csv' or f['path'].startswith('tracking/')]
    for entry in needed:
        path = raw / entry['path']
        if not path.exists() and args.download:
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(f'https://raw.githubusercontent.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data/{REVISION}/data/{entry["path"]}', path)
        if not path.exists():
            raise FileNotFoundError(f'{path}: rerun with --download')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
    plays = pd.read_csv(raw / 'plays.csv')
    assert not plays.duplicated(['gameId', 'playId']).any()
    labeled = plays.loc[plays.pff_passCoverage.notna()]
    snapshots = []
    for entry in needed:
        if not entry['path'].startswith('tracking/'):
            continue
        df = pd.read_csv(raw / entry['path'], usecols=TRACKING + ['event'])
        snaps = df.loc[df.event.eq('ball_snap')].groupby(['gameId', 'playId']).frameId.min().rename('snapFrame')
        df = df.join(snaps, on=['gameId', 'playId'])
        pre = df.loc[df.frameId.lt(df.snapFrame) & df.nflId.notna() & df.x.notna() & df.y.notna()]
        pre = pre.merge(labeled[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']], on=['gameId', 'playId'], validate='many_to_one')
        for _, rows in pre.groupby(['gameId', 'playId'], sort=True):
            teams = rows.team.eq(rows.possessionTeam) | rows.team.eq(rows.defensiveTeam)
            rows = rows.assign(isDefense=rows.team.eq(rows.defensiveTeam).astype(int), validTeam=teams)
            stats = rows.groupby('frameId').agg(n=('nflId', 'size'), unique=('nflId', 'nunique'), defense=('isDefense', 'sum'), valid=('validTeam', 'all'))
            valid = stats.loc[stats.n.eq(22) & stats.unique.eq(22) & stats.defense.eq(11) & stats.valid]
            if valid.empty:
                continue
            frame = valid.index.max()
            snapshots.append(rows.loc[rows.frameId.eq(frame), TRACKING + ['snapFrame', 'isDefense']])
    tracking = pd.concat(snapshots, ignore_index=True).sort_values(['gameId', 'playId', 'nflId'])
    tracking['nflId'] = tracking.nflId.astype('int64')
    tracking['snapFrame'] = tracking.snapFrame.astype('int64')
    keys = tracking[['gameId', 'playId']].drop_duplicates()
    selected = labeled.merge(keys, on=['gameId', 'playId'], validate='one_to_one').sort_values(['gameId', 'playId'])
    assert tracking.frameId.lt(tracking.snapFrame).all()
    assert tracking.groupby(['gameId', 'playId']).size().eq(22).all()
    assert tracking.groupby(['gameId', 'playId']).isDefense.sum().eq(11).all()
    assert not tracking.duplicated(['gameId', 'playId', 'nflId']).any()
    selected[CONTEXT].to_csv(out / 'plays.csv', index=False)
    selected[['gameId', 'playId', 'pff_passCoverage', 'pff_passCoverageType']].to_csv(out / 'labels.csv', index=False)
    tracking.to_csv(out / 'tracking.csv', index=False)
    excluded = plays.merge(keys.assign(included=True), on=['gameId', 'playId'], how='left')
    excluded = excluded.loc[excluded.included.isna(), ['gameId', 'playId', 'pff_passCoverage']]
    excluded['reason'] = excluded.pff_passCoverage.map(lambda x: 'missing coverage label' if pd.isna(x) else 'no complete strictly pre-snap 11-v-11 frame')
    excluded.to_csv(out / 'excluded_plays.csv', index=False)
    summary = {'source': SOURCE, 'revision': REVISION, 'source_plays': len(plays),
               'included_plays': len(selected), 'excluded_plays': len(excluded), 'tracking_rows': len(tracking),
               'coverage_counts': selected.pff_passCoverage.value_counts().to_dict(),
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
