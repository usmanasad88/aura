"""Tests for defect detector.

Tests detection of composite layup defects.
"""

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DefectType(Enum):
    """Types of defects detectable in composite layup."""
    DRY_SPOT = "dry_spot"
    AIR_BUBBLE = "air_bubble"
    WRINKLE = "wrinkle"
    FIBER_MISALIGNMENT = "fiber_misalignment"
    RESIN_RICH = "resin_rich"
    RESIN_STARVED = "resin_starved"


@dataclass
class MockBoundingBox:
    """Bounding box for defect location."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass
class MockDefect:
    """Detected defect in layup."""
    type: DefectType
    bbox: MockBoundingBox
    confidence: float
    severity: str
    remedy: str
    area_mm2: float


DEFECT_DEFINITIONS = {
    DefectType.DRY_SPOT: {
        "description": "Unwetted fiberglass area - appears white/opaque",
        "severity": "high",
        "remedy": "Apply additional resin and work in with brush or roller"
    },
    DefectType.AIR_BUBBLE: {
        "description": "Trapped air between plies - visible as raised area",
        "severity": "medium",
        "remedy": "Roll toward edge to expel air"
    },
    DefectType.WRINKLE: {
        "description": "Fold or crease in fiberglass ply",
        "severity": "high",
        "remedy": "Lift ply carefully, smooth out, and re-wet if needed"
    },
    DefectType.FIBER_MISALIGNMENT: {
        "description": "Fibers not oriented in specified direction",
        "severity": "medium",
        "remedy": "Reposition ply before resin cures"
    },
    DefectType.RESIN_RICH: {
        "description": "Excessive resin pooling",
        "severity": "low",
        "remedy": "Use squeegee to remove excess resin"
    },
    DefectType.RESIN_STARVED: {
        "description": "Insufficient resin - fibers visible and dry",
        "severity": "high",
        "remedy": "Apply additional resin immediately"
    },
}


class MockDefectDetector:
    """Mock defect detector for testing."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
        self.min_defect_area_mm2 = config.get("min_defect_area_mm2", 25) if config else 25
        self._mock_detections: list[MockDefect] = []
    
    def add_mock_detection(self, defect: MockDefect) -> None:
        """Add a mock detection for testing."""
        self._mock_detections.append(defect)
    
    def detect(self, image=None, ply_number: int = 1) -> list[MockDefect]:
        """Detect defects in image (mock implementation)."""
        # Filter by confidence and area thresholds
        return [
            d for d in self._mock_detections
            if d.confidence >= self.confidence_threshold
            and d.area_mm2 >= self.min_defect_area_mm2
        ]
    
    def get_remedy(self, defect_type: DefectType) -> str:
        """Get remedy for a defect type."""
        return DEFECT_DEFINITIONS.get(defect_type, {}).get("remedy", "Unknown remedy")
    
    def get_severity(self, defect_type: DefectType) -> str:
        """Get severity for a defect type."""
        return DEFECT_DEFINITIONS.get(defect_type, {}).get("severity", "unknown")


class TestDefectTypes:
    """Test defect type definitions."""
    
    def test_all_types_defined(self):
        """Ensure all defect types have definitions."""
        for defect_type in DefectType:
            assert defect_type in DEFECT_DEFINITIONS
            assert "severity" in DEFECT_DEFINITIONS[defect_type]
            assert "remedy" in DEFECT_DEFINITIONS[defect_type]
    
    def test_high_severity_defects(self):
        """Test which defects are high severity."""
        high_severity = [
            DefectType.DRY_SPOT,
            DefectType.WRINKLE,
            DefectType.RESIN_STARVED,
        ]
        for defect_type in high_severity:
            assert DEFECT_DEFINITIONS[defect_type]["severity"] == "high"
    
    def test_medium_severity_defects(self):
        """Test which defects are medium severity."""
        medium_severity = [
            DefectType.AIR_BUBBLE,
            DefectType.FIBER_MISALIGNMENT,
        ]
        for defect_type in medium_severity:
            assert DEFECT_DEFINITIONS[defect_type]["severity"] == "medium"


class TestDefectDetector:
    """Test defect detection functionality."""
    
    def test_no_detections(self):
        """Test empty result when no defects."""
        detector = MockDefectDetector()
        defects = detector.detect()
        assert defects == []
    
    def test_single_detection(self):
        """Test detecting a single defect."""
        detector = MockDefectDetector()
        
        defect = MockDefect(
            type=DefectType.DRY_SPOT,
            bbox=MockBoundingBox(100, 100, 150, 150),
            confidence=0.85,
            severity="high",
            remedy="Apply additional resin",
            area_mm2=100
        )
        detector.add_mock_detection(defect)
        
        detections = detector.detect()
        assert len(detections) == 1
        assert detections[0].type == DefectType.DRY_SPOT
    
    def test_confidence_filtering(self):
        """Test that low confidence detections are filtered."""
        detector = MockDefectDetector({"confidence_threshold": 0.7})
        
        # High confidence - should pass
        detector.add_mock_detection(MockDefect(
            type=DefectType.AIR_BUBBLE,
            bbox=MockBoundingBox(0, 0, 50, 50),
            confidence=0.8,
            severity="medium",
            remedy="Roll out bubble",
            area_mm2=100
        ))
        
        # Low confidence - should be filtered
        detector.add_mock_detection(MockDefect(
            type=DefectType.WRINKLE,
            bbox=MockBoundingBox(100, 100, 150, 150),
            confidence=0.5,
            severity="high",
            remedy="Smooth wrinkle",
            area_mm2=100
        ))
        
        detections = detector.detect()
        assert len(detections) == 1
        assert detections[0].type == DefectType.AIR_BUBBLE
    
    def test_area_filtering(self):
        """Test that small detections are filtered."""
        detector = MockDefectDetector({"min_defect_area_mm2": 50})
        
        # Large enough - should pass
        detector.add_mock_detection(MockDefect(
            type=DefectType.DRY_SPOT,
            bbox=MockBoundingBox(0, 0, 100, 100),
            confidence=0.9,
            severity="high",
            remedy="Apply resin",
            area_mm2=100
        ))
        
        # Too small - should be filtered
        detector.add_mock_detection(MockDefect(
            type=DefectType.AIR_BUBBLE,
            bbox=MockBoundingBox(200, 200, 210, 210),
            confidence=0.9,
            severity="medium",
            remedy="Ignore small bubble",
            area_mm2=20
        ))
        
        detections = detector.detect()
        assert len(detections) == 1
        assert detections[0].type == DefectType.DRY_SPOT
    
    def test_multiple_defects(self):
        """Test detecting multiple defects."""
        detector = MockDefectDetector()
        
        defects = [
            MockDefect(DefectType.DRY_SPOT, MockBoundingBox(0, 0, 50, 50), 0.9, "high", "Wet out", 100),
            MockDefect(DefectType.AIR_BUBBLE, MockBoundingBox(100, 0, 130, 30), 0.8, "medium", "Roll", 50),
            MockDefect(DefectType.WRINKLE, MockBoundingBox(0, 100, 80, 120), 0.85, "high", "Smooth", 75),
        ]
        
        for d in defects:
            detector.add_mock_detection(d)
        
        detections = detector.detect()
        assert len(detections) == 3
    
    def test_get_remedy(self):
        """Test getting remedy for defect type."""
        detector = MockDefectDetector()
        
        remedy = detector.get_remedy(DefectType.DRY_SPOT)
        assert "resin" in remedy.lower()
        
        remedy = detector.get_remedy(DefectType.AIR_BUBBLE)
        assert "roll" in remedy.lower()
    
    def test_get_severity(self):
        """Test getting severity for defect type."""
        detector = MockDefectDetector()
        
        assert detector.get_severity(DefectType.DRY_SPOT) == "high"
        assert detector.get_severity(DefectType.AIR_BUBBLE) == "medium"
        assert detector.get_severity(DefectType.RESIN_RICH) == "low"


class TestBoundingBox:
    """Test bounding box functionality."""
    
    def test_area_calculation(self):
        """Test bounding box area calculation."""
        bbox = MockBoundingBox(0, 0, 10, 10)
        assert bbox.area == 100
    
    def test_non_square_area(self):
        """Test non-square bounding box."""
        bbox = MockBoundingBox(10, 20, 30, 60)
        assert bbox.area == 20 * 40  # 800
    
    def test_offset_box(self):
        """Test box not at origin."""
        bbox = MockBoundingBox(100, 200, 150, 280)
        assert bbox.area == 50 * 80  # 4000
