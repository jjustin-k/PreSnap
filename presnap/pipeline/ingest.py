import csv
import math
from pathlib import Path
from statistics import median
from collections.abc import Iterator, Mapping
from os import PathLike

play_to_coverage: dict[tuple[str, str], str] = {}
play_to_tracking: dict[tuple[str, str], list[dict[str, str]]] = {}

ALIGNMENT_FIELDS = [
    'defense_deepest_depth', 'defense_second_deepest_depth',
    'defense_deep_pair_lateral_spacing', 'defense_width',
    'defenders_beyond_5_yards', 'defenders_beyond_10_yards',
    'defenders_beyond_15_yards', 'offense_width',
    'offense_left_count', 'offense_right_count', 'offense_backfield_count',
    'offense_wide_count', 'wide_offense_nearest_defender_mean',
    'wide_offense_nearest_defender_min', 'wide_offense_nearest_defender_max',
    *[f'defender_{rank}_{axis}' for rank in range(1, 12)
      for axis in ('depth', 'lateral')],
]


fields: list[str] = [
    'game_id',
    'offensive_team',
    'defensive_team',
    'quarter',
    'game_clock',
    'score_gap',
    'down',
    'yards_to_fd',
    'yards_to_goal',
    'offensive_line',
    'wide_recievers',
    'running_backs',
    'tight_ends',
    'formation',
    'defensive_line',
    'line_backers',
    'defensive_backs',
    'defenders_in_box',
    *ALIGNMENT_FIELDS,
    'defensive_coverage',
]

ACTIVE_PLAYERS_ON_FIELD: int = 11

def assign_labels(file_path: str | PathLike[str]) -> None:
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            play_to_coverage[(row['gameId'], row['playId'])] = row['pff_passCoverage']

def distance_to_goal(offense: str, yardline_side: str, yardline_number: str | int) -> int:
    if offense == yardline_side:
        return 100 - int(yardline_number)
    return int(yardline_number)

def extract_play_info(file_path: str | PathLike[str]) -> Iterator[dict[str, str]]:
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            yield row

def assign_tracking(file_path: str | PathLike[str]) -> None:
    """Index player snapshots by game AND play; do not use row order as identity."""
    play_to_tracking.clear()
    for row in extract_play_info(file_path):
        key = (row['gameId'], row['playId'])
        play_to_tracking.setdefault(key, []).append(row)


def alignment_features(data: Mapping[str, str]) -> dict[str, float | int]:
    """Describe one complete pre-snap snapshot, in yards and offense-relative axes.

    Positive depth is downfield from the line of scrimmage. Rotate leftward
    plays 180 degrees, then center lateral coordinates on the offense's median
    lateral position (a proxy for formation center, NOT the football).
    Wide players are >5 yards from this center; backfield players are >1 yard
    behind the line. These are geometric categories, not player position labels.
    """
    rows = play_to_tracking.get((data['gameId'], data['playId']), [])
    if len(rows) != 22 or len({r['nflId'] for r in rows}) != 22:
        raise ValueError('Expected 22 distinct tracked players')
    if len({r['frameId'] for r in rows}) != 1 or len({r['snapFrame'] for r in rows}) != 1:
        raise ValueError('Expected a single snapshot and snap frame')
    if any(int(r['frameId']) >= int(r['snapFrame']) for r in rows):
        raise ValueError('Tracking must be strictly before the snap')
    directions = {r['playDirection'] for r in rows}
    if len(directions) != 1 or not directions <= {'left', 'right'}:
        raise ValueError('Invalid or inconsistent play direction')
    sign = 1 if rows[0]['playDirection'] == 'right' else -1
    # Derive the line when the source coordinate is missing.
    distance = distance_to_goal(data['possessionTeam'], data['yardlineSide'], data['yardlineNumber'])
    los = float(data['absoluteYardlineNumber']) if data.get('absoluteYardlineNumber') else (
        110 - distance if sign == 1 else 10 + distance
    )
    offense, defense = [], []
    for row in rows:
        side = int(row['isDefense'])
        if side not in (0, 1):
            raise ValueError('Invalid isDefense flag')
        expected_team = data['defensiveTeam'] if side else data['possessionTeam']
        if row['team'] != expected_team:
            raise ValueError('Tracking team does not match play context')
        depth, lateral = sign * (float(row['x']) - los), sign * float(row['y'])
        if not all(math.isfinite(v) for v in (depth, lateral)):
            raise ValueError('Non-finite player coordinate')
        (defense if side else offense).append((depth, lateral))
    if len(offense) != 11 or len(defense) != 11:
        raise ValueError('Expected 11 players per team')
    center = median(y for _, y in offense)
    offense = [(x, y - center) for x, y in offense]
    defense = sorted(((x, y - center) for x, y in defense), key=lambda xy: (-xy[0], xy[1]))
    wide = [(x, y) for x, y in offense if abs(y) > 5]
    distances = [min(math.hypot(x - dx, y - dy) for dx, dy in defense) for x, y in wide]
    result = {
        'defense_deepest_depth': defense[0][0],
        'defense_second_deepest_depth': defense[1][0],
        'defense_deep_pair_lateral_spacing': abs(defense[0][1] - defense[1][1]),
        'defense_width': max(y for _, y in defense) - min(y for _, y in defense),
        **{f'defenders_beyond_{threshold}_yards': sum(x > threshold for x, _ in defense)
           for threshold in (5, 10, 15)},
        'offense_width': max(y for _, y in offense) - min(y for _, y in offense),
        'offense_left_count': sum(y > 0 for _, y in offense),
        'offense_right_count': sum(y < 0 for _, y in offense),
        'offense_backfield_count': sum(x < -1 for x, _ in offense),
        'offense_wide_count': len(wide),
        # Zero distances mean no wide players only when offense_wide_count == 0.
        'wide_offense_nearest_defender_mean': sum(distances) / len(distances) if distances else 0.0,
        'wide_offense_nearest_defender_min': min(distances, default=0.0),
        'wide_offense_nearest_defender_max': max(distances, default=0.0),
    }
    for rank, (depth, lateral) in enumerate(defense, start=1):
        result[f'defender_{rank}_depth'] = depth
        result[f'defender_{rank}_lateral'] = lateral
    return result


def transform_data(data: Mapping[str, str]) -> dict[str, str | int | float]:
    raw_personal = data['personnelO'].split(' ')
    qb_on_field = 1

    if len(raw_personal) == 4:
        oline_on_field = int(raw_personal[0])
        rb_on_field = int(raw_personal[2])
        te_on_field = int(raw_personal[4])
        wr_on_field = int(raw_personal[6])
    else:
        rb_on_field = int(raw_personal[0])
        te_on_field = int(raw_personal[2])
        wr_on_field = int(raw_personal[4])

        oline_on_field = ACTIVE_PLAYERS_ON_FIELD - rb_on_field - te_on_field - wr_on_field - qb_on_field


    raw_personal = data['personnelD'].split(' ')
    dl_on_field = int(raw_personal[0])
    lb_on_field = int(raw_personal[2])
    db_on_field = int(raw_personal[4])

    endzone_distance = distance_to_goal(data['possessionTeam'], data['yardlineSide'], data['yardlineNumber'])
    minutes, seconds = data["gameClock"].split(":")
    game_clock = int(minutes) * 60 + int(seconds)

    return {
        'game_id' : data['gameId'],
        # Game situation
        'offensive_team' : data['possessionTeam'],
        'defensive_team' : data['defensiveTeam'],
        'quarter' : int(data['quarter']),
        'game_clock' : game_clock,
        'score_gap' : abs(int(data['preSnapHomeScore']) - int(data['preSnapVisitorScore'])),


        'down' : int(data['down']),
        'yards_to_fd' : int(data['yardsToGo']),
        'yards_to_goal' : endzone_distance,

        # Offensive formations
        'offensive_line' : oline_on_field,
        'wide_recievers' : wr_on_field,
        'running_backs' : rb_on_field,
        'tight_ends' : te_on_field,
        'formation' : data['offenseFormation'],

        # Defensive formations
        'defensive_line' : dl_on_field,
        'line_backers' : lb_on_field,
        'defensive_backs' : db_on_field,
        'defenders_in_box': data['defendersInBox'],

        # Pre-snap player geometry
        **alignment_features(data),

        #Target
        'defensive_coverage' : play_to_coverage[(data['gameId'],data['playId'])]
    }

def validate_transformation(cleaned_data) -> bool:
    for v in cleaned_data.values():
        if v is None or v == '':
            return False
    return True


def load(file_path: str | PathLike[str], cleaned_data: Mapping[str, str | int | float]) -> None:
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cleaned_data.keys())
        writer.writerow(cleaned_data)


if __name__ == '__main__':

    output_path = 'data/cleaned_data/cleaned_data.csv'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    assign_tracking('data/presnap/tracking.csv')

    with open(output_path, mode='w', encoding='utf-8') as f:
        f.write(",".join(fields) + '\n')


    assign_labels('data/presnap/labels.csv')

    written = skipped = 0
    for play in extract_play_info('data/presnap/plays.csv'):
        try:
            cleaned_data = transform_data(play)
            if validate_transformation(cleaned_data):
                load(output_path, cleaned_data)
                written += 1
            else:
                skipped += 1
        except ValueError as e:
            skipped += 1
            print(f"Skipped {play['gameId']}/{play['playId']}: {e}")
    print(f"Wrote {written} plays; skipped {skipped}; added {len(ALIGNMENT_FIELDS)} alignment features.")

