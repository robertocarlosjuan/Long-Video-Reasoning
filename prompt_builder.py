import os

class PromptBuilder:
    def __init__(self, config):
        self.style = config.prompt_style
        self.max_pixels = config.max_edge_len * config.max_edge_len
        # Sampling strategy parameters
        self.sampling_strategy = getattr(config, 'sampling_strategy', 'fps')  # 'fps' or 'coarse'
        self.fps = config.sampling_rate  # Default fps when using fps strategy
        self.num_frames = getattr(config, 'num_frames', 8)  # Default num_frames when using coarse strategy

    def num_turns(self):
        if self.style == "stage_analysis":
            return 4
        elif self.style == "coarse_cot":
            return 4
        elif self.style == "frame_labeling":
            return 3
        return 2
    
    def uses_query(self, turn_index):
        if self.style == "stage_analysis":
            return turn_index == 2
        if self.style == "frame_labeling":
            return turn_index == 1
        return True

    def build(self, query=None, video_path=None, timestamps=[], segments=None, prior_response=None, turn_index=0):
        # Helper function to create video content with appropriate sampling strategy
        def create_video_content(path, max_pixels, segment=None):
            content = {
                "type": "video",
                "video": f"{path}",
                "max_pixels": max_pixels,
            }
            if self.sampling_strategy == 'fps':
                content["fps"] = self.fps
            else:  # coarse sampling
                content["num_frames"] = self.num_frames
            
            # Add segment timestamps if provided
            if segment:
                content["start_time"] = segment[0]
                content["end_time"] = segment[1]
            
            return content

        if self.style == "baseline":
            return 'answer', 1, [
                    {
                        "role": "user",
                        "content": [
                            create_video_content(video_path, self.max_pixels),
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
                            create_video_content(video_path, self.max_pixels),
                            {"type": "text", "text": f"Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage."},
                        ]
                    }
                ]
            elif turn_index == 1:
                label_string = ", ".join(timestamps)
                stages = '\n'.join([f"{timestamps[i]} - {timestamps[i+1]}: [Stage] -> [Stage]" for i in range(len(timestamps)-1)])
                return 'segmentation', 1,[
                    {
                        "role": "user",
                        "content": [
                            create_video_content(video_path, self.max_pixels),
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
                video_segments = [create_video_content(video_path, self.max_pixels, segment) for segment in segments]
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": video_segments + [
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
                            create_video_content(video_path, self.max_pixels),
                            {"type": "text", "text": f"""Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage."""},
                        ]
                    }
                ]
            elif turn_index == 1:
                label_string = ", ".join(timestamps)
                stages = '\n'.join([f"{timestamps[i]} - {timestamps[i+1]}: [Stage] -> [Stage]" for i in range(len(timestamps)-1)])
                return 'segmentation', 5,[
                    {
                        "role": "user",
                        "content": [
                            create_video_content(video_path, self.max_pixels),
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
Crucially, for the final "Verdict" line in each segment, only one of the words "Relevant" or "Irrelevant" should appear. No additional text should be included there.
Present your answer in the following format:
Required evidence: [Brief Description of required evidence to answer the query]
Segment Relevance:
{segment_explanations}
"""}]
                    }
                ]
            elif turn_index == 3:
                video_segments = [create_video_content(video_path, self.max_pixels, segment) for segment in segments]
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": video_segments + [
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
                            create_video_content(video_path, self.max_pixels),
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
                video_segments = [create_video_content(video_path, self.max_pixels, segment) for segment in segments]
                return 'answer', 1, [
                    {
                        "role": "user",
                        "content": video_segments + [
                            {"type": "text", "text": f"Based on the video, answer the question. Do not use bounding boxes. Give only textual descriptions. If the answer cannot be determined from available information, say 'Not enough information'. Do not use full sentences. Give only the answer.\nQuestion:{query}"},
                        ]
                    }
                ]
        else:
            raise ValueError("Unsupported prompt style")