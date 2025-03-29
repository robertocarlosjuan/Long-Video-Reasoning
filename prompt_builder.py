class PromptBuilder:
    def __init__(self, config):
        self.style = config.prompt_style

    def num_turns(self):
        if self.style == "stage_analysis":
            return 3
        if self.style == "frame_labeling":
            return 2
        return 1
    
    def uses_query(self, turn_index):
        if self.style == "stage_analysis":
            return turn_index == 2
        if self.style == "frame_labeling":
            return turn_index == 1
        return True

    def build(self, query=None, frames_paths=None, timestamps=None, prior_response=None, turn_index=0):
        if self.style == "default":
            return [{"role": "user", "content": f"Describe what is happening. Query: {query}"}]

        elif self.style == "instruction":
            return [{"role": "user", "content": f"Instruction: Describe what is happening.\nUser: {query}"}]

        elif self.style == "chat":
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the person doing in the video?"},
                {"role": "user", "content": query},
            ]

        elif self.style == "stage_analysis":
            if turn_index == 0:
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths},
                            {"type": "text", "text": f"Using the provided video frames, determine the overall scenario depicted and express it as a short phrase. Next, list the common process stages associated with that scenario. For each stage, provide a concise stage name along with a brief descriptive phrase that captures the essence of that stage."},
                        ]
                    }
                ]
            elif turn_index == 1:
                label_string = ", ".join(timestamps or [])
                stages = '\n'.join([f"{timestamps[i]} - {timestamps[i+1]}: [Stage] -> [Stage]" for i in range(len(timestamps)-1)])
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths},
                            {"type": "text", "text": f"Given that the frames respond to the timestamps {label_string}, for each video segment between each frames, determine the stage transitions. \nStages are as described here:\n{prior_response}\nPresent your answer in the following format:\n{stages}"""},
                        ]
                    }
                ]
            elif turn_index == 2:
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {query}\nBriefly describe what kind of scene would clearly show the answer to the query.\nGiven the following video segments:\n{prior_response}\nDetermine which segment(s) are most likely to contain the answer to the question.\nPresent your answer in the following format:\n[Brief Description of expected scene]\n[start_time] - [end_time]\nIf multiple segments are applicable, list each one on a separate line."},
                        ]
                    }
                ]

        elif self.style == "frame_labeling":
            label_string = ", ".join(timestamps[:3] + ["... up to "] + [timestamps[-1]] or [])
            if turn_index == 0:
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames_paths},
                            {"type": "text", "text": f"The provided frames are sampled from a video at 1 fps, with timestamps from {label_string}. Your task is to:\n\n1. Determine the overall scenario depicted in the video and express it as a short phrase.\n2. Identify and describe each process stage occurring throughout the video. Ensure that the stages cover the entire video duration from {timestamps[0]} to {timestamps[-1]} without any gaps.\n3. For each stage, provide:\n   - A concise stage name.\n   - A brief descriptive phrase that captures the essence of that stage.\n   - The corresponding start and end timestamps as whole numbers (since the frames are sampled at 1 fps).\n\nPresent your answer using the exact following format:\n\n[Overall Scenario]\n[start_time] - [end_time]: [Stage name]: [Brief Description of Stage]"},
                        ]
                    }
                ]
            elif turn_index == 1:
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {query}\nBriefly describe what kind of scene would clearly show the answer to the query.\nGiven the following video segments:\n{prior_response}\nDetermine which segment(s) are most likely to contain the answer to the question.\nPresent your answer in the following format:\n[Brief Description of expected scene]\n[start_time] - [end_time]\nIf multiple segments are applicable, list each one on a separate line."},
                        ]
                    }
                ]

        else:
            raise ValueError("Unsupported prompt style")