class Story():
    def __init__(self, id, name, s1, s2, s3, s4, e_correct):
        self.id = id
        self.name = name
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.e_correct = e_correct

    def get_story_as_list(self):
        return [self.s1, self.s2, self.s3, self.s4, self.e_correct]