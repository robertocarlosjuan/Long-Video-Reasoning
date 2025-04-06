import re
import math
from nltk.metrics.distance import edit_distance

def select_best_by_edit_distance(candidates):
    """
    Given a list of text candidates, returns the one that has
    the lowest total edit distance when compared with all others.
    
    :param candidates: List of strings
    :return: String (the best candidate)
    """
    if not candidates:
        return None
    
    min_sum_distance = math.inf
    best_answer = None
    
    for i, cand in enumerate(candidates):
        total_distance = 0
        for j, other in enumerate(candidates):
            if i != j:
                total_distance += edit_distance(cand, other)
        if total_distance < min_sum_distance:
            min_sum_distance = total_distance
            best_answer = cand
    
    return best_answer

def parse_stages(prior_text):
    """
    Extract stages from PRIOR_RESPONSE, returning a list of valid stage names.
    """
    valid_stages = []
    lines = prior_text.splitlines()
    for line in lines:
        line = line.strip()
        # Look for lines like "1. Identifying faulty components"
        match = re.match(r"^\d+\.\s*(.*)$", line)
        if match:
            stage_name = match.group(1).strip().lower()
            stage_name = re.sub(r'[^a-z]', '', stage_name)
            valid_stages.append(stage_name)
    return valid_stages



def check_segments(response_text, valid_stages):
    """
    Check each segment in RESPONSE to see if the before/after stages match valid stages.
    """
    lines = response_text.splitlines()
    segment_pattern = re.compile(r"^(.*?)\:\s*(.*?)\s*->\s*(.*?)\s*$")
    before_valid = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = segment_pattern.match(line)
        if match:
            # 1. e.g., "0s - 128s" (not used for checking, just context)
            segment_time = match.group(1).strip()
            # 2. e.g., "Identifying faulty components"
            before_stage = match.group(2).strip().lower()
            before_stage = re.sub(r'[^a-z]', '', before_stage)
            # 3. e.g., "Disconnecting circuit breakers"
            after_stage = match.group(3).strip().lower()
            after_stage = re.sub(r'[^a-z]', '', after_stage)
            before_valid = any(valid_stage in before_stage for valid_stage in valid_stages)
            after_valid = any(valid_stage in after_stage for valid_stage in valid_stages)

            if not before_valid:
                return False
            if not after_valid:
                return False
    return before_valid

def check_segmentation_validity(response_text, prior_response):
    """
    Check if the segmentation is valid.
    """
    valid_stages = parse_stages(prior_response)
    return check_segments(response_text, valid_stages)

