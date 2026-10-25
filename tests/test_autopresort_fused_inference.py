from auto_presort.fused_inference import _build_track, build_window_classifier


def test_dashboard_states_map_to_existing_pet_four_classes():
    per_frame = [
        [{"track_id": 1, "final_state": "EATING", "model_probs": {"eat": 0.8}}],
        [{"track_id": 1, "final_state": "ACTIVE", "model_probs": {"other": 0.7}}],
        [{"track_id": 1, "final_state": "RESTING", "model_probs": {"other": 0.9}}],
    ]
    classify = build_window_classifier(
        "pet", per_frame, 1.0, ["eat", "drink", "active", "resting"]
    )

    assert classify(0, 2) == ("eat", 0.8)
    assert classify(1, 3) == ("active", 0.7)


def test_dashboard_track_output_builds_review_track():
    result = {
        "video": {"fps": 2.0},
        "frame_size": {"width": 640, "height": 480},
        "results": [
            {"left": {"detections": [{
                "track_id": 1, "bbox": [10, 20, 100, 120],
                "final_state": "ACTIVE", "model_probs": {"active": 0.8},
            }]}},
            {"left": {"detections": [{
                "track_id": 2, "bbox": [200, 20, 300, 120],
                "final_state": "RESTING", "model_probs": {"other": 0.9},
            }, {
                "track_id": 1, "bbox": [11, 20, 101, 120],
                "final_state": "ACTIVE", "model_probs": {"active": 0.8},
            }]}},
        ],
    }

    track, per_frame = _build_track("pet", result)

    assert track["frame_w"] == 640
    assert track["frame_h"] == 480
    assert per_frame[0][0]["track_id"] == 1
    assert per_frame[1][0]["track_id"] == 1
    assert track["flags"] == [True, True]
