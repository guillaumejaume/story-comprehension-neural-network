class Story():
    def __init__(self, id, s1, s2, s3, s4, e_right, e_wrong = ""):
        self.id = id
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.e_right = e_right
        self.e_wrong = e_wrong

    def get_story_with_right_ending_as_list(self):
        return [self.s1, self.s2, self.s3, self.s4, self.e_right]

    def get_story_with_both_endings_as_list(self):
        return [self.s1, self.s2, self.s3, self.s4, self.e_right, self.e_wrong]