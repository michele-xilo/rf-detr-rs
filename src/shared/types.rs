use std::fmt::Display;

pub struct MeanStdNormalizationConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Bbox {
    BboxXYXY(BboxXYXY),
    BboxXYWH(BboxXYWH),
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BboxXYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BboxXYWH {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
}

impl From<BboxXYXY> for BboxXYWH {
    fn from(item: BboxXYXY) -> Self {
        let center_x = (item.x1 + item.x2) / 2.0;
        let center_y = (item.y1 + item.y2) / 2.0;
        let width = (item.x2 - item.x1).abs();
        let height = (item.y2 - item.y1).abs();
        Self {
            center_x,
            center_y,
            width,
            height,
        }
    }
}

impl From<BboxXYWH> for BboxXYXY {
    fn from(item: BboxXYWH) -> Self {
        let half_width = item.width / 2.0;
        let half_height = item.height / 2.0;
        Self {
            x1: item.center_x - half_width,
            y1: item.center_y - half_height,
            x2: item.center_x + half_width,
            y2: item.center_y + half_height,
        }
    }
}

impl Bbox {
    pub fn to_xyxy(&self) -> BboxXYXY {
        match self {
            Bbox::BboxXYXY(bbox) => *bbox,
            Bbox::BboxXYWH(bbox) => (*bbox).into(),
        }
    }

    pub fn to_xywh(&self) -> BboxXYWH {
        match self {
            Bbox::BboxXYWH(bbox) => *bbox,
            Bbox::BboxXYXY(bbox) => (*bbox).into(),
        }
    }

    pub fn calculate_iou(&self, other: &Bbox) -> f32 {
        let det1 = self.to_xyxy();
        let det2 = other.to_xyxy();

        let x1_1 = det1.x1;
        let y1_1 = det1.y1;
        let x2_1 = det1.x2;
        let y2_1 = det1.y2;
        let x1_2 = det2.x1;
        let y1_2 = det2.y1;
        let x2_2 = det2.x2;
        let y2_2 = det2.y2;

        // Calculate intersection area
        let inter_x1 = x1_1.max(x1_2);
        let inter_y1 = y1_1.max(y1_2);
        let inter_x2 = x2_1.min(x2_2);
        let inter_y2 = y2_1.min(y2_2);
        let inter_area = if inter_x2 > inter_x1 && inter_y2 > inter_y1 {
            (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        } else {
            0.0
        };

        // Calculate union area
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        let union_area = area1 + area2 - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Landmark2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug)]
pub enum ModelSize {
    Nano,
    Lite,
    Medium,
    Full,
    Large,
    Heavy,
}

impl Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSize::Nano => write!(f, "Nano"),
            ModelSize::Lite => write!(f, "Lite"),
            ModelSize::Medium => write!(f, "Medium"),
            ModelSize::Full => write!(f, "Full"),
            ModelSize::Large => write!(f, "Large"),
            ModelSize::Heavy => write!(f, "Heavy"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameterized::parameterized;

    #[test]
    fn test_iou_same_format() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYXY(BboxXYXY {
            x1: 5.0,
            y1: 5.0,
            x2: 15.0,
            y2: 15.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert!((iou - 0.1428).abs() < 0.001);
    }

    #[test]
    fn test_iou_different_formats() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 10.0,
            center_y: 10.0,
            width: 10.0,
            height: 10.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert!((iou - 0.1428).abs() < 0.001);
    }

    #[test]
    fn test_iou_no_overlap() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYXY(BboxXYXY {
            x1: 20.0,
            y1: 20.0,
            x2: 30.0,
            y2: 30.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_iou_identical() {
        let bbox1 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 5.0,
            center_y: 5.0,
            width: 10.0,
            height: 10.0,
        });
        let bbox2 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 5.0,
            center_y: 5.0,
            width: 10.0,
            height: 10.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert_eq!(iou, 1.0);
    }

    #[test]
    fn test_xyxy_xywh_round_trip() {
        // Arrange
        let xyxy = BboxXYXY {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 60.0,
        };

        // Act
        let round_trip = Bbox::BboxXYWH(Bbox::BboxXYXY(xyxy).to_xywh()).to_xyxy();

        // Assert
        assert_eq!(round_trip, xyxy);
    }

    #[parameterized(
        model_size = {ModelSize::Nano, ModelSize::Lite, ModelSize::Medium, ModelSize::Full, ModelSize::Large, ModelSize::Heavy},
        expected_path = {"Nano", "Lite", "Medium", "Full", "Large", "Heavy"}
    )]
    fn test_model_size_display(model_size: ModelSize, expected_path: &str) {
        assert_eq!(model_size.to_string(), expected_path);
    }
}
