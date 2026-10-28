// Source of truth: rfdetr/assets/coco_classes.py + rfdetr/datasets/coco.py.
//
// The model's classification head emits 91 slots. Training uses raw COCO
// `category_id` values directly as labels (coco.py:271-272), and COCO ids start
// at 1 — so slot 0 is unused (DETR no-object), and slots 1..=90 map directly to
// COCO category ids (with gaps at 12, 26, 29, 30, 45, 66, 68, 69, 71, 83 left
// empty and filtered downstream).
pub const COCO_CLASS_NAMES: [&str; 91] = [
    "",               // 0 (no-object slot)
    "person",         //  1
    "bicycle",        //  2
    "car",            //  3
    "motorcycle",     //  4
    "airplane",       //  5
    "bus",            //  6
    "train",          //  7
    "truck",          //  8
    "boat",           //  9
    "traffic light",  // 10
    "fire hydrant",   // 11
    "",               // 12 (gap)
    "stop sign",      // 13
    "parking meter",  // 14
    "bench",          // 15
    "bird",           // 16
    "cat",            // 17
    "dog",            // 18
    "horse",          // 19
    "sheep",          // 20
    "cow",            // 21
    "elephant",       // 22
    "bear",           // 23
    "zebra",          // 24
    "giraffe",        // 25
    "",               // 26 (gap)
    "backpack",       // 27
    "umbrella",       // 28
    "",               // 29 (gap)
    "",               // 30 (gap)
    "handbag",        // 31
    "tie",            // 32
    "suitcase",       // 33
    "frisbee",        // 34
    "skis",           // 35
    "snowboard",      // 36
    "sports ball",    // 37
    "kite",           // 38
    "baseball bat",   // 39
    "baseball glove", // 40
    "skateboard",     // 41
    "surfboard",      // 42
    "tennis racket",  // 43
    "bottle",         // 44
    "",               // 45 (gap)
    "wine glass",     // 46
    "cup",            // 47
    "fork",           // 48
    "knife",          // 49
    "spoon",          // 50
    "bowl",           // 51
    "banana",         // 52
    "apple",          // 53
    "sandwich",       // 54
    "orange",         // 55
    "broccoli",       // 56
    "carrot",         // 57
    "hot dog",        // 58
    "pizza",          // 59
    "donut",          // 60
    "cake",           // 61
    "chair",          // 62
    "couch",          // 63
    "potted plant",   // 64
    "bed",            // 65
    "",               // 66 (gap)
    "dining table",   // 67
    "",               // 68 (gap)
    "",               // 69 (gap)
    "toilet",         // 70
    "",               // 71 (gap)
    "tv",             // 72
    "laptop",         // 73
    "mouse",          // 74
    "remote",         // 75
    "keyboard",       // 76
    "cell phone",     // 77
    "microwave",      // 78
    "oven",           // 79
    "toaster",        // 80
    "sink",           // 81
    "refrigerator",   // 82
    "",               // 83 (gap)
    "book",           // 84
    "clock",          // 85
    "vase",           // 86
    "scissors",       // 87
    "teddy bear",     // 88
    "hair drier",     // 89
    "toothbrush",     // 90
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_length_matches_logit_space() {
        // Assert
        assert_eq!(COCO_CLASS_NAMES.len(), 91);
    }

    #[test]
    fn test_known_indices_are_one_indexed_by_coco_id() {
        // Assert — slot index == coco id
        assert!(COCO_CLASS_NAMES[0].is_empty(), "slot 0 must be no-object");
        assert_eq!(COCO_CLASS_NAMES[1], "person");
        assert_eq!(COCO_CLASS_NAMES[2], "bicycle");
        assert_eq!(COCO_CLASS_NAMES[62], "chair");
        assert_eq!(COCO_CLASS_NAMES[63], "couch");
        assert_eq!(COCO_CLASS_NAMES[90], "toothbrush");
    }

    #[test]
    fn test_gap_indices_are_empty() {
        // Assert — coco id is the slot index directly
        for gap_id in [12_usize, 26, 29, 30, 45, 66, 68, 69, 71, 83] {
            assert!(
                COCO_CLASS_NAMES[gap_id].is_empty(),
                "expected gap at coco id / slot {gap_id}"
            );
        }
    }
}
