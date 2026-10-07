# Traffic-light timetable

The benchmark's signal timetable. The launcher drives the 13 scenes it lists with it
(`tlc_timetable_set=tlc_timetable`, `tlc_timetable_sets_dir=OdysseyBenchmark/data`, plus
`tl_control_path=<the scene's tl_control.json>`; see `OdysseyTrafficAgent/odyssey/manager/tlc_timetable_set.py`,
`default_runner.yaml`) and scores their designated signals against it (`tl_set=tlc_timetable`).
Default `null` = no change.

| file | what |
|---|---|
| `tlc_timetable_set.json` | manifest: the 13 scenes and, per scene, `allow_excluded` / `allow_unobserved` |
| `<scene>.json` (13) | the scene's signal timetable, `tl_signal_patch/1` (GT rows 0..999, R/G spans per connector) |
| `designated.json` | the signals scored by the timetable rule (12 signals in 11 scenes) |

The representative-frame labels (`tl_control/1`) are not in this directory: each scene uses its own
published `tl_control.json`. Every run pins the manifest's sha256.

Scenes: TLC-scored odyssey_scene011, 025, 032, 052, 056, 060, 061, 067, 072, 075, 086; a fixed
constant timetable at every signalised ego-route intersection in odyssey_scene021 and 081.
`allow_excluded`: odyssey_scene011, 021, 056, 081 (on `tl_control.TL_CONTROL_SCENE_EXCLUSIONS`).
`allow_unobserved`: odyssey_scene060.
