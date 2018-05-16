class Story():
    def __init__(self, id, s1, s2, s3, s4, e_correct, e_incorrect):
        self.id = id
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.e_correct = e_correct
        self.e_incorrect = e_incorrect

    @property
    def id(self):
        """ Getter for id
           Returns:
           --------
           id: string
           """
        return self.id

    @id.setter
    def id(self, value):
        """ Setter for id
            Parameters:
            -----------
            value: string
            id
            """
        self.id = value

    @property
    def s1(self):
        """ Getter for s1
            Returns:
            --------
            s1: string
            First sentence
        """
        return self.s1

    @s1.setter
    def s1(self, value):
        """ Setter for s1
            Parameters:
            -----------
            value: string
            First sentence
            """
        self.s1 = value

    @property
    def s2(self):
        """ Getter for s2
            Returns:
            --------
            s2: string
            Second sentence
        """
        return self.s2

    @s2.setter
    def s2(self, value):
        """ Setter for s2
            Parameters:
            -----------
            value: string
            Second sentence
            """
        self.s2 = value

    @property
    def s3(self):
        """ Getter for s3
            Returns:
            --------
            s3: string
            Third sentence
        """
        return self.s3

    @s3.setter
    def s3(self, value):
        """ Setter for s3
            Parameters:
            -----------
            value: string
            Third sentence
            """
        self.s3 = value

    @property
    def s4(self):
        """ Getter for s4
            Returns:
            --------
            s4: string
            Forth sentence
        """
        return self.s4

    @s4.setter
    def s4(self, value):
        """ Setter for s4
            Parameters:
            -----------
            value: string
            Forth sentence
            """
        self.s4 = value

    @property
    def e_correct(self):
        """ Getter for e_correct
            Returns:
            --------
            e_correct: string
            Correct end of story
        """
        return self.e_correct

    @e_correct.setter
    def e_correct(self, value):
        """ Setter for e_correct
            Parameters:
            -----------
            value: string
            Correct end of story
            """
        self.e_correct = value

    @property
    def e_incorrect(self):
        """ Getter for e_incorrect
                    Returns:
                    --------
                    e_incorrect: string
                    Incorrect end of story
        """
        return self.e_incorrect

    @e_incorrect.setter
    def e_incorrect(self, value):
        """ Setter for e_incorrect
            Parameters:
            -----------
            value: string
            Incorrect end of story
            """
        self.e_incorrect = value