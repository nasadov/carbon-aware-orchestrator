#!/usr/bin/env python3
"""
Test script to validate the fix for timeslot validation.
This test verifies that pods cannot be scheduled in past timeslots.
"""

import sys
import logging
from datetime import datetime, timedelta

# Add the server-python directory to the path so we can import carbon_aware modules
sys.path.append("/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python")

from carbon_aware.models import CarbonAwarePod, CarbonAwareTimeslot
from carbon_aware.utils import is_timeslot_valid

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def test_timeslot_validity():
    """Test the is_timeslot_valid function with different scenarios."""
    now = datetime.now()
    print(f"Current time: {now}")
    
    # Create a pod with a deadline 12 hours from now
    pod = CarbonAwarePod(
        id="test-pod",
        deadline_hours=12.0,  # 12 hours from now
        duration=2.0,
        powerConsumption=0.1,
        cpuRequest=1.0,
        ramRequest=512.0,
        storageRequest=1024
    )
    print(f"Pod deadline: {pod.deadline}")
    
    # Test cases
    
    # Case 1: Current hour timeslot (should be valid)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    current_ts = CarbonAwareTimeslot(
        id=0, 
        startYear=current_hour.year,
        startMonth=current_hour.month,
        startDay=current_hour.day,
        startHour=current_hour.hour,
        length=1
    )
    valid = is_timeslot_valid(current_ts, pod)
    print(f"\nCase 1 - Current hour timeslot: {current_ts.getStart()} to {current_ts.getEnd()}")
    print(f"Valid: {valid} (expect: True)")
    
    # Case 2: Past hour timeslot (should be invalid)
    past_hour = now - timedelta(hours=1)
    past_hour = past_hour.replace(minute=0, second=0, microsecond=0)
    past_ts = CarbonAwareTimeslot(
        id=1,
        startYear=past_hour.year,
        startMonth=past_hour.month,
        startDay=past_hour.day,
        startHour=past_hour.hour,
        length=1
    )
    valid = is_timeslot_valid(past_ts, pod)
    print(f"\nCase 2 - Past hour timeslot: {past_ts.getStart()} to {past_ts.getEnd()}")
    print(f"Valid: {valid} (expect: False)")
    
    # Case 3: Future hour timeslot within deadline (should be valid)
    future_hour = now + timedelta(hours=2)
    future_hour = future_hour.replace(minute=0, second=0, microsecond=0)
    future_ts = CarbonAwareTimeslot(
        id=2,
        startYear=future_hour.year,
        startMonth=future_hour.month,
        startDay=future_hour.day,
        startHour=future_hour.hour,
        length=1
    )
    valid = is_timeslot_valid(future_ts, pod)
    print(f"\nCase 3 - Future hour timeslot: {future_ts.getStart()} to {future_ts.getEnd()}")
    print(f"Valid: {valid} (expect: True)")
    
    # Case 4: Future hour timeslot beyond deadline (should be invalid)
    beyond_deadline = now + timedelta(hours=15)
    beyond_deadline = beyond_deadline.replace(minute=0, second=0, microsecond=0)
    beyond_ts = CarbonAwareTimeslot(
        id=3,
        startYear=beyond_deadline.year,
        startMonth=beyond_deadline.month,
        startDay=beyond_deadline.day,
        startHour=beyond_deadline.hour,
        length=1
    )
    valid = is_timeslot_valid(beyond_ts, pod)
    print(f"\nCase 4 - Beyond deadline timeslot: {beyond_ts.getStart()} to {beyond_ts.getEnd()}")
    print(f"Valid: {valid} (expect: False)")
    
    # Case 5: Timeslot that has already ended (should be invalid)
    past_ended = now - timedelta(hours=2)
    past_ended = past_ended.replace(minute=0, second=0, microsecond=0)
    past_ended_ts = CarbonAwareTimeslot(
        id=4,
        startYear=past_ended.year,
        startMonth=past_ended.month,
        startDay=past_ended.day,
        startHour=past_ended.hour,
        length=1
    )
    valid = is_timeslot_valid(past_ended_ts, pod)
    print(f"\nCase 5 - Timeslot that has already ended: {past_ended_ts.getStart()} to {past_ended_ts.getEnd()}")
    print(f"Valid: {valid} (expect: False)")


if __name__ == "__main__":
    test_timeslot_validity()
