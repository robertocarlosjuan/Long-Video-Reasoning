class PromptBuilder:
    def __init__(self, config):
        self.style = config.prompt_style
        self.max_pixels = config.max_edge_len * config.max_edge_len

    def num_turns(self):
        if self.style == "stage_analysis":
            return 4
        if self.style == "frame_labeling":
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
                return 'segmentation', 5,[
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
                            {"type": "text", "text": f"Question: {query}\nBriefly describe what kind of scene would clearly show the answer to the query.\nGiven the following video segments:\n{prior_response}\nDetermine which segment(s) are most likely to contain the answer to the question.\nPresent your answer in the following format:\n[Brief Description of expected scene]\n[start_time] - [end_time]\nIf multiple segments are applicable, list each one on a separate line."},
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