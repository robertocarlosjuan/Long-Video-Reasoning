class PromptBuilder:
    def __init__(self, config):
        self.style = config.prompt_style
        self.max_pixels = config.max_edge_len * config.max_edge_len

    def num_turns(self):
        if self.style == "stage_analysis":
            return 4
        elif self.style == "coarse_cot":
            return 4
        elif self.style == "coarse_cot_shot_detection":
            return 2
        elif self.style == "frame_labeling":
            return 3
        return 2
    
    def uses_query(self, turn_index):
        if self.style == "stage_analysis":
            return turn_index == 2
        if self.style == "frame_labeling":
            return turn_index == 1
        return True

    def build(self, query=None, frames_paths=None, timestamps=None, selected_frames_list = [],prior_response=None, turn_index=0):
        if self.style == "baseline":
            return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"}
                        ]
                    }
                ]

        elif self.style == "instruction":
            return 'answer', 1, [{"role": "user", "content": f"Instruction: Describe what is happening.\nUser: {query}"}]

        elif self.style == "chat":
            return 'answer', 1,[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the person doing in the video?"},
                {"role": "user", "content": query},
            ]

        elif self.style == "stage_analysis":
            if turn_index == 0:
                return 'understanding', 1,[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels  },
                            {"type": "text", "text": f"Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage."},
                        ]
                    }
                ]
            elif turn_index == 1:
                label_string = ", ".join(timestamps or [])
                stages = '\n'.join([f"{timestamps[i]} - {timestamps[i+1]}: [Stage] -> [Stage]" for i in range(len(timestamps)-1)])
                return 'segmentation', 1,[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Given that the frames respond to the timestamps {label_string}, for each video segment between each frames, determine the stage transitions. \nStages are as described here:\n{prior_response}\nPresent your answer strictly in the following format:\n{stages}\nDo not include any explanations, descriptions, or additional text. Only provide the stage transitions in the exact format above."""},
                        ]
                    }
                ]
            elif turn_index == 2:
                return 'segment_selection', 5, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {query}\nBriefly describe what kind of scene would clearly show the answer to the query.\nGiven the following video segments:\n{prior_response}\nDetermine which segment(s) are most likely to contain the answer to the question.\nPresent your answer in the following format:\n[Brief Description of expected scene]\n[start_time] - [end_time]: [Explanation of why this segment is relevant]\nIf multiple segments are applicable, list each one on a separate line."},
                        ]
                    }
                ]
            elif turn_index == 3:
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": selected_frames_list, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"},
                        ]
                    }
                ]
        elif self.style == "coarse_cot":
            if turn_index == 0:
                return 'understanding', 1,[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"""Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage."""},
                        ]
                    }
                ]
            elif turn_index == 1:
                label_string = ", ".join(timestamps or [])
                stages = '\n'.join([f"{timestamps[i]} - {timestamps[i+1]}: [Stage] -> [Stage]" for i in range(len(timestamps)-1)])
                return 'segmentation', 5,[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"""Given that the frames respond to the timestamps {label_string}, for each video segment between each frames, determine the stage transitions.
Stages are as described here:
{prior_response}
Present your answer strictly in the following format:
{stages}
Do not include any explanations, descriptions, or additional text. Only provide the stage transitions in the exact format above."""},
                        ]
                    }
                ]
            elif turn_index == 2:
                segment_explanations = '\n'.join([f"{i+1}. {timestamps[i]} - {timestamps[i+1]}:   \nRelevant: [Explanation on how this segment could be relevant]\nIrrelevant: [Explanation on how this segment could be irrelevant]\nConfidence Score: [Confidence score from 0 to 10]\nVerdict: [Relevant/Irrelevant]" for i in range(len(timestamps)-1)])
                return 'segment_selection', 5, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"""Question: {query}
Determine the required evidence to answer the query.
Given the following video segments:
{prior_response}
For each segment, explain how the segment could be relevant to the query, how the segment could be irrelevant to the query, give a confidence score and a final verdict.
Ground your explanations in the video segment stages above, focusing on how the segment relates to the query.
Use varying confidence scores to reflect each segment's true relevance or irrelevance,ensuring at least one receives a score of 0 and at least one receives a score of 10.
Follow the output format strictly and do not give additional explanations. 
Crucially, for the final “Verdict” line in each segment, only one of the words “Relevant” or “Irrelevant” should appear. No additional text should be included there.
Present your answer in the following format:
Required evidence: [Brief Description of required evidence to answer the query]
Segment Relevance:
{segment_explanations}
"""}]
                    }
                ]
            elif turn_index == 3:
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": selected_frames_list, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"},
                        ]
                    }
                ]
            
        elif self.style == "coarse_cot_shot_detection":
#             if turn_index == 0:
#                 return 'understanding', 1,[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
#                             {"type": "text", "text": f"""Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage.
# Output format:
# Overall Scenario: [Short phrase]
# Stages:
# 1. [Stage name] - [Short description]
# 2. [Stage name] - [Short description]
# 3. [Stage name] - [Short description]
# 4. [Stage name] - [Short description]
# """},
#                         ]
#                     }
#                 ]
#             elif turn_index == 1:
#                 label_string = ", ".join([f"{timestamps[i]}" for i in range(len(timestamps))] or [])
#                 stages = '\n'.join([f"{timestamps[i]}: [Stage name] - [Based on the frame, a short description of what could be happening in this video segment]" for i in range(len(timestamps))])
#                 return 'segmentation', 5,[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
#                             {"type": "text", "text": f"""Given that the frames correspond to the video segments {label_string}, for each video segment, determine the stage and based on the frame, give a short description of what could be happening in this video segment.
# Stages are as described here:
# {prior_response}
# Present your answer strictly in the following format:
# {stages}
# Do not include any explanations, descriptions, or additional text. Only provide the stage transitions in the exact format above."""},
#                         ]
#                     }
#                 ]
            if turn_index == 0:
                label_string = ", ".join([f"{timestamps[i]}" for i in range(len(timestamps))] or [])
                segment_explanations = '\n'.join([f"{i+1}. {timestamps[i]}:   \nDescription: [Description of what could be happening in this video segment]\nRelevant: [Explanation on how this segment could be relevant]\nIrrelevant: [Explanation on how this segment could be irrelevant]\nConfidence Score: [Confidence score from 0 to 10]\nVerdict: [Relevant/Irrelevant]" for i in range(len(timestamps))])
                return 'segment_selection', 5, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"""Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. 
Question: {query}
Determine the required evidence to answer the query.
Each frame represents a key moment from a video segment. The timestamp range of each segment is overlaid directly on the top-left corner of the image. Use it as context when determining what the segment depicts and how it relates to the query.
For each segment, describe what could be happening in the segment and explain how the segment could be relevant to the query, how the segment could be irrelevant to the query, give a confidence score and a final verdict.
Ground your explanations in the description of what could be happening in the segment, focusing on how the segment relates to the query.
Use varying confidence scores to reflect each segment's true relevance or irrelevance,ensuring at least one receives a score of 0 and at least one receives a score of 10.
Follow the output format strictly and do not give additional explanations. 
Crucially, for the final “Verdict” line in each segment, only one of the words “Relevant” or “Irrelevant” should appear. No additional text should be included there.
Present your answer in the following format, filling in the square brackets with your answer:
Overall Scenario: [Short description of the overall scenario]
Required evidence: [Brief Description of required evidence to answer the query]
Segment Relevance:
{segment_explanations}
"""}]
                    }
                ]
            elif turn_index == 1:
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": selected_frames_list, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"},
                        ]
                    }
                ]

        
        elif self.style == "frame_labeling":
            label_string = ", ".join(timestamps[:3] + ["... up to "] + [timestamps[-1]] or [])
            if turn_index == 0:
                return 'segmentation', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"The provided frames are sampled from a video at 1 fps, with timestamps from {timestamps[0]} to {timestamps[-1]}. Your task is to analyze the full video and:\n\n1. Determine the overall scenario in a short phrase.\n2. Divide the video into process stages that span the entire duration without gaps or overlaps.\n3. For each stage, output only the following format:\n\n[start_time] - [end_time]: [Stage name]\n\nYour final output should start with:\n\n[Overall Scenario]: [Short phrase]\n\nFollowed by the stage breakdown lines in the specified format."},
                        ]
                    }
                ]
            elif turn_index == 1:
                return 'segment_selection', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {query}\nBriefly describe what kind of scene would clearly show the answer to the query.\nGiven the following video segments:\n{prior_response}\nDetermine which segment(s) are most likely to contain the answer to the question.\nPresent your answer in the following format:\n[Brief Description of expected scene]\n[start_time] - [end_time]\nIf multiple segments are applicable, list each one on a separate line."},
                        ]
                    }
                ]
            elif turn_index == 2:
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": selected_frames_list, "max_pixels": self.max_pixels},
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"},
                        ]
                    }
                ]
        else:
            raise ValueError("Unsupported prompt style")