import unittest
from presnap.pipeline import ingest


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.previous = ingest.play_to_tracking.copy()
        self.play = dict(gameId='1', playId='2', possessionTeam='A', defensiveTeam='B',
                         yardlineSide='A', yardlineNumber='40', absoluteYardlineNumber='50')
        self.rows = [dict(nflId=str(i), frameId='9', snapFrame='10', playDirection='right',
                          isDefense=str(i // 11), team='A' if i < 11 else 'B',
                          x=str(48.25 if i < 11 else 51.25 + (i-11)),
                          y=str(10.25 + (i % 11)*3)) for i in range(22)]
        ingest.play_to_tracking[('1', '2')] = self.rows

    def tearDown(self):
        ingest.play_to_tracking.clear()
        ingest.play_to_tracking.update(self.previous)

    def test_geometry_and_schema(self):
        values = ingest.alignment_features(self.play)
        self.assertEqual(set(values), set(ingest.ALIGNMENT_FIELDS))
        self.assertEqual(values['defense_deepest_depth'], 11.25)
        self.assertEqual(values['defense_second_deepest_depth'], 10.25)
        self.assertEqual(values['defenders_beyond_10_yards'], 2)
        self.assertEqual(values['defense_width'], 30)
        self.assertEqual(values['offense_backfield_count'], 11)
        self.assertEqual(values['offense_wide_count'], 8)

    def test_row_order_invariance(self):
        expected = ingest.alignment_features(self.play)
        ingest.play_to_tracking[('1', '2')] = list(reversed(self.rows))
        self.assertEqual(expected, ingest.alignment_features(self.play))

    def test_rotation_invariance(self):
        expected = ingest.alignment_features(self.play)
        self.play['absoluteYardlineNumber'] = '70'
        ingest.play_to_tracking[('1', '2')] = [dict(r, playDirection='left',
            x=str(120-float(r['x'])), y=str(160/3-float(r['y']))) for r in self.rows]
        actual = ingest.alignment_features(self.play)
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key], places=9, msg=key)

    def test_missing_line_fallback(self):
        expected = ingest.alignment_features(self.play)
        self.play['absoluteYardlineNumber'] = ''
        self.assertEqual(expected, ingest.alignment_features(self.play))

    def test_rejects_snap_frame_and_incomplete_snapshot(self):
        self.rows[0]['frameId'] = '10'
        with self.assertRaises(ValueError):
            ingest.alignment_features(self.play)
        self.rows.pop()
        with self.assertRaises(ValueError):
            ingest.alignment_features(self.play)

    def test_empty_wide_group_is_finite(self):
        for row in self.rows:
            row['y'] = '25'
        values = ingest.alignment_features(self.play)
        self.assertEqual(values['offense_wide_count'], 0)
        self.assertEqual(values['wide_offense_nearest_defender_mean'], 0)


if __name__ == '__main__':
    unittest.main()
