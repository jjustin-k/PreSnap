import csv
from collections.abc import Iterator, Mapping
from os import PathLike

play_to_coverage: dict[tuple[str, str], str] = {}

fields: list[str] = [
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
    'defensive_coverage',
]

ACTIVE_PLAYERS_ON_FIELD: int = 11

def assign_labels(file_path: str | PathLike[str]) -> None:
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            play_to_coverage[(row['gameId'], row['playId'])] = f"{row['pff_passCoverage']}-{row['pff_passCoverageType']}"

def distance_to_goal(offense: str, yardline_side: str, yardline_number: str | int) -> int:
    if offense == yardline_side:
        return 100 - int(yardline_number)
    return int(yardline_number)

def extract_play_info(file_path: str | PathLike[str]) -> Iterator[dict[str, str]]:
    with open(file_path, 'r') as f:
        for row in csv.DictReader(f):
            yield row

def transform_data(data: Mapping[str, str]) -> dict[str, str | int]:
    raw_personal = data['personnelO'].split(' ')
    rb_on_field = int(raw_personal[0])
    te_on_field = int(raw_personal[2])
    wr_on_field = int(raw_personal[4])
    qb_on_field = 1
    oline_on_field = ACTIVE_PLAYERS_ON_FIELD - rb_on_field - te_on_field - wr_on_field - qb_on_field


    raw_personal = data['personnelD'].split(' ')
    dl_on_field = int(raw_personal[0])
    lb_on_field = int(raw_personal[2])
    db_on_field = int(raw_personal[4])

    endzone_distance = distance_to_goal(data['possessionTeam'], data['yardlineSide'], data['yardlineNumber'])

    return {

        # Game situation
        'offensive_team' : data['possessionTeam'],
        'defensive_team' : data['defensiveTeam'],
        'quarter' : int(data['quarter']),
        'game_clock' : data['gameClock'],
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

        #Target
        'defensive_coverage' : play_to_coverage[(data['gameId'],data['playId'])] 
    }

def load(file_path: str | PathLike[str], cleaned_data: Mapping[str, str | int]) -> None:
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cleaned_data.keys())
        writer.writerow(cleaned_data)


if __name__ == '__main__':

    output_path = 'data/cleaned_data/cleaned_data.csv'

    with open(output_path, mode='w', encoding='utf-8') as f:
        f.write(",".join(fields) + '\n')


    assign_labels('data/presnap/labels.csv')

    for play in extract_play_info('data/presnap/plays.csv'):
        try:
            cleaned_data = transform_data(play)
            load(output_path, cleaned_data)
        except ValueError:
            pass

