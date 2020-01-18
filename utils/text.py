import codecs


class TextFeaturizer(object):
    """Extract text feature based on char-level granularity.

       By looking up the vocabulary table, each input string (one line of transcript)
       will be converted to a sequence of integer indexes.
    """

    def __init__(self, characters_file):
        lines = []
        with codecs.open(characters_file, "r", "utf-8") as f:
            lines.extend(f.readlines())

        self.token_to_index = {}
        self.index_to_token = {}
        self.speech_labels = ""
        index = 0

        for line in lines:
            line = line[:-1]
            if line.startswith("#"):
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.speech_labels += line
            index += 1
