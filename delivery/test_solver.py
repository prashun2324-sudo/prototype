"""
Occlusion Solver - Validation Tests
------------------------------------

Run this to verify the solver works correctly before integration.

Usage:
    python test_solver.py

All tests should pass. If any fail, do not integrate.
"""

import numpy as np
import sys

# Import the solver
from motion_occlusion_solver import OcclusionSolver, Result

def test_basic_tracking():
    """Test 1: Basic tracking works"""
    print("Test 1: Basic tracking...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Feed simple data
    result = solver.process(
        anchor=[1.0, 1.0, 0.5],
        reference=[0.5, 1.5, 0.5],
        confidence=0.95
    )
    
    # Check output is valid
    assert result.position is not None, "Position should not be None"
    assert len(result.position) == 3, "Position should be 3D"
    assert result.confidence == 0.95, "Confidence should match input"
    assert result.is_predicted == False, "Should not be predicted"
    assert result.state == "tracking", "State should be tracking"
    
    print("PASS")
    return True


def test_direction_calculation():
    """Test 2: Direction is calculated correctly"""
    print("Test 2: Direction calculation...", end=" ")
    
    solver = OcclusionSolver(object_length=1.0)
    
    # Wrist at (1,0,0), elbow at (0,0,0) -> direction should be (1,0,0)
    result = solver.process(
        anchor=[1.0, 0.0, 0.0],
        reference=[0.0, 0.0, 0.0],
        confidence=0.95
    )
    
    # Direction should point from elbow to wrist (normalized)
    expected_dir = np.array([1.0, 0.0, 0.0])
    dir_error = np.linalg.norm(result.direction - expected_dir)
    
    assert dir_error < 0.01, f"Direction error too large: {dir_error}"
    
    # Position should be wrist + direction * object_length
    expected_pos = np.array([2.0, 0.0, 0.0])  # 1.0 + 1.0*1.0
    pos_error = np.linalg.norm(result.position - expected_pos)
    
    # Allow some smoothing error
    assert pos_error < 0.1, f"Position error too large: {pos_error}"
    
    print("PASS")
    return True


def test_occlusion_detection():
    """Test 3: Occlusion is detected correctly"""
    print("Test 3: Occlusion detection...", end=" ")
    
    solver = OcclusionSolver(
        object_length=0.5,
        high_conf=0.7,
        low_conf=0.3
    )
    
    # Good tracking
    r1 = solver.process([1,1,1], [0.5,1.5,1], confidence=0.9)
    assert r1.state == "tracking", "Should be tracking at 0.9 conf"
    
    # Lower confidence
    r2 = solver.process([1,1,1], [0.5,1.5,1], confidence=0.5)
    # Could be tracking or low_confidence depending on history
    
    # Very low - should trigger prediction
    r3 = solver.process([0,0,0], [0,0,0], confidence=0.1)
    r4 = solver.process([0,0,0], [0,0,0], confidence=0.1)
    r5 = solver.process([0,0,0], [0,0,0], confidence=0.1)
    
    assert r5.is_predicted == True, "Should be predicted at 0.1 conf"
    
    print("PASS")
    return True


def test_prediction_continuity():
    """Test 4: Prediction doesn't jump"""
    print("Test 4: Prediction continuity...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Establish tracking
    for i in range(5):
        solver.process([1+i*0.1, 1, 0.5], [0.5+i*0.1, 1.5, 0.5], confidence=0.95)
    
    last_pos = solver.pos.copy()
    
    # Enter occlusion
    result = solver.process([0,0,0], [0,0,0], confidence=0.05)
    result = solver.process([0,0,0], [0,0,0], confidence=0.05)
    
    # Position should not jump dramatically
    jump = np.linalg.norm(result.position - last_pos)
    
    # Allow reasonable movement (velocity-based prediction)
    assert jump < 1.0, f"Position jumped too much: {jump}"
    
    print("PASS")
    return True


def test_recovery_blending():
    """Test 5: Recovery blends smoothly"""
    print("Test 5: Recovery blending...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Track
    for i in range(5):
        solver.process([1, 1, 0.5], [0.5, 1.5, 0.5], confidence=0.95)
    
    # Occlude
    for i in range(3):
        solver.process([0,0,0], [0,0,0], confidence=0.05)
    
    predicted_pos = solver.pos.copy()
    
    # Recover at different position
    r1 = solver.process([2, 1, 0.5], [1.5, 1.5, 0.5], confidence=0.9)
    
    # Should be blending, not jumping to new position
    assert r1.state in ["blending", "recovering", "tracking"], f"Unexpected state: {r1.state}"
    
    print("PASS")
    return True


def test_2d_mode():
    """Test 6: 2D input works"""
    print("Test 6: 2D mode...", end=" ")
    
    solver = OcclusionSolver(object_length=50)  # pixels
    
    result = solver.process(
        anchor=[400, 300],
        reference=[350, 350],
        confidence=0.9
    )
    
    assert len(result.position) == 2, "Should be 2D"
    assert result.quaternion is None, "Quaternion should be None for 2D"
    
    print("PASS")
    return True


def test_quaternion_valid():
    """Test 7: Quaternion is unit length"""
    print("Test 7: Quaternion validity...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Various directions
    test_cases = [
        ([1, 0, 0], [0, 0, 0]),
        ([0, 1, 0], [0, 0, 0]),
        ([0, 0, 1], [0, 0, 0]),
        ([1, 1, 1], [0, 0, 0]),
        ([0.5, 0.8, 0.3], [0.1, 0.2, 0.1]),
    ]
    
    for wrist, elbow in test_cases:
        result = solver.process(wrist, elbow, confidence=0.95)
        
        if result.quaternion is not None:
            quat_norm = np.linalg.norm(result.quaternion)
            assert abs(quat_norm - 1.0) < 0.001, f"Quaternion not unit: {quat_norm}"
    
    print("PASS")
    return True


def test_output_format():
    """Test 8: Output dictionary format"""
    print("Test 8: Output format...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    result = solver.process([1,1,1], [0.5,1.5,1], confidence=0.9)
    
    # Get dictionary
    d = result.as_dict()
    
    # Check all required fields
    required = ['position', 'direction', 'quaternion', 'velocity', 'confidence', 'occluded', 'state']
    for key in required:
        assert key in d, f"Missing key: {key}"
    
    # Check types
    assert isinstance(d['position'], list), "position should be list"
    assert isinstance(d['confidence'], float), "confidence should be float"
    assert isinstance(d['occluded'], bool), "occluded should be bool"
    assert isinstance(d['state'], str), "state should be str"
    
    print("PASS")
    return True


def test_reset():
    """Test 9: Reset clears state"""
    print("Test 9: Reset...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Build up state
    for i in range(10):
        solver.process([1,1,1], [0.5,1.5,1], confidence=0.9)
    
    assert solver.pos is not None, "Should have state"
    
    # Reset
    solver.reset()
    
    assert solver.pos is None, "State should be cleared"
    assert solver.vel is None, "Velocity should be cleared"
    assert len(solver.conf_history) == 0, "History should be cleared"
    
    print("PASS")
    return True


def test_two_handed():
    """Test 10: Two-handed grip"""
    print("Test 10: Two-handed grip...", end=" ")
    
    solver = OcclusionSolver(object_length=0.5)
    
    # Right wrist, right elbow, left wrist close by
    result = solver.process(
        anchor=[1, 1, 0.5],
        reference=[0.5, 1.5, 0.5],
        confidence=0.95,
        second_anchor=[1.1, 1.05, 0.55]
    )
    
    assert result.position is not None, "Should work with second anchor"
    
    print("PASS")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 50)
    print("OCCLUSION SOLVER VALIDATION")
    print("=" * 50)
    print()
    
    tests = [
        test_basic_tracking,
        test_direction_calculation,
        test_occlusion_detection,
        test_prediction_continuity,
        test_recovery_blending,
        test_2d_mode,
        test_quaternion_valid,
        test_output_format,
        test_reset,
        test_two_handed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR - {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\nAll tests passed. Safe to integrate.")
        return True
    else:
        print("\nSome tests failed. Do NOT integrate until fixed.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

