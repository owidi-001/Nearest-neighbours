"""
COMP 614
Homework 5: Bag of Words
"""

import re

import numpy
import string

import comp614_module5


def get_title_and_text(filename):
    """
    Given a name of an XML file, extracts and returns the strings contained
    between the <title></title> and <text></text> tags.
    """
    title_pattern = r'(?<=\<title\>)([\s\S]*?)(?=\<\/title\>)'
    text_pattern = r'<text[^>]*>([\s\S]*?)<\/text>'
    # Use context manager to open and read files
    with open(filename, mode='r', encoding="utf-8") as file:
        file_content = file.read()

        file_content = file_content.replace('\n', '')

        result_titles = ""
        result_texts = ""

        # Use regex to get all title and texts
        titles = re.findall(title_pattern, file_content)

        texts = re.findall(text_pattern, file_content)

        # Append titles and text contents to the result_titles and results texts respectively
        for title in titles:
            result_titles += (title + ' ')

        for text in texts:
            result_texts += (text + ' ')

        file.close()

    # Return the result_titles and texts
    return result_titles.strip(), result_texts.strip()


def get_words(text):
    """
    Given the full text of an XML file, filters out the non-body text (text that
    is contained within {{}}, [[]], [], <>, etc.) and punctuation and returns a
    list of the remaining words, each of which should be converted to lowercase.
    """

    temp = text.lower()

    # Manually replace all occurrences
    temp = re.sub(r"({{)[\s\S]*?(}})", '', temp)
    temp = re.sub(r"({)[\s\S]*?(})", '', temp)
    temp = re.sub(r"({\|)[\s\S]*?(\|})", '', temp)
    temp = re.sub(r"(\[\[)[\s\S]*?(]])", '', temp)
    temp = re.sub(r"(\[)[\s\S]*?(])", '', temp)
    temp = re.sub(r"(<)[\s\S]*?(>)", '', temp)
    temp = re.sub(r"(&lt;)[\s\S]*?(&gt;)", '', temp)
    # temp = re.sub(r":\|", '', temp)
    temp = re.sub(r":.*?:\|", '', temp)

    temp = temp.replace(r'\t', ' ')

    punctuation_to_remove = "[" + string.punctuation + "](?![st]\\s)"

    temp = re.sub(punctuation_to_remove, ' ', temp)

    return temp.split()


def count_words(words):
    """
    Given a list of words, returns the total number of words as well as a
    dictionary mapping each unique word to its frequency of occurrence.
    """
    word_count = {}
    # unq_words = set(words)  # getting unique words
    # word_to_dict = {word: words.count(word) for word in unq_words}
    for word in words:
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] += 1

    # return len(set(words)), word_count
    return len(words), word_count


def count_all_words(filenames):
    """
    Given a list of filenames, returns three things. First, a list of the titles,
    where the i-th title corresponds to the i-th input filename. Second, a
    dictionary mapping each filename to an inner dictionary mapping each unique
    word in that file to its relative frequency of occurrence. Last, a dictionary
    mapping each unique word --- including all words found across all files ---
    to its total frequency of occurrence across all the input files.
    """

    all_titles = []
    title_to_counter = {}
    total_counts = {}

    # Iterate through the files given
    for filename in filenames:
        title, text = get_title_and_text(filename)
        all_titles.append(title)  # Add titles

        # get count
        total_words, word_count = count_words(get_words(text))

        # normalize
        for key, value in word_count.items():

            # Normalize and round off to
            word_count[key] = (value / total_words)

            # update word value in total counts
            if key in total_counts.keys():
                total_counts[key] += value
            else:
                total_counts[key] = value

        # add word count to title counter
        title_to_counter[title] = word_count

    return all_titles, title_to_counter, total_counts


def encode_word_counts(all_titles, title_to_counter, total_counts, num_words):
    """
    Given two dictionaries in the format output by count_all_words and an integer
    num_words representing the number of top words to encode, finds the top
    num_words words in total_counts and builds a matrix where the element in
    position (i, j) is the relative frequency of occurrence of the j-th most
    common overall word in the i-th article (i.e., the article corresponding to
    the i-th title in titles).
    """
    if all_titles:
        pass

    sorted_words = sorted(total_counts.items(), key=lambda tup: (- 1 * tup[1], tup[0]))
    top_k_words = sorted_words[:num_words]

    top_k_counter = [[] for _ in range((len(top_k_words)))]

    top_words = numpy.zeros((len(title_to_counter), len(top_k_words)))

    counter = 0

    for filename in title_to_counter:
        for top_word in range(len(top_k_words)):
            top_k_counter[top_word] = title_to_counter[filename].get(top_k_words[top_word][0])

            if top_k_counter[top_word] is None:
                top_k_counter[top_word] = 0

        top_words[counter] = top_k_counter

        counter += 1

    return numpy.matrix(top_words)


def nearest_neighbors(matrix, all_titles, title, num_neighbours):
    """
    Given a matrix, a list of all titles whose data is encoded in the matrix, such
    that the i-th title corresponds to the i-th row, a single title whose data is
    encoded in the matrix, and the desired number of neighbors to be found, finds
    and returns the closest neighbors to the article with the given title.
    """

    title_num = all_titles.index(title)
    distance_mat = []
    for val in range(matrix.shape[0]):
        distance = numpy.sqrt(numpy.sum(numpy.square(matrix[title_num] - matrix[val])))
        distance_mat.append(distance)

    nearest_nbr = numpy.argsort(distance_mat)

    nearest_nbr_titles = [all_titles[title] for title in nearest_nbr]

    k_nearest_nbr_titles = nearest_nbr_titles[1: (num_neighbours + 1)]

    return k_nearest_nbr_titles


def run():
    """
    Encodes the wikipedia dataset into a matrix, prompts the user to choose an
    article, and then runs the knn algorithm to find the 5 nearest neighbors
    of the chosen article.
    """
    # Encode the wikipedia dataset in a matrix
    # filenames = comp614_module5.ALL_FILES
    filenames = comp614_module5.ALL_FILES
    all_titles, title_to_counter, total_counts = count_all_words(filenames)
    mat = encode_word_counts(all_titles, title_to_counter, total_counts, 20000)

    # Print all articles
    print("Enter the integer corresponding to the article whose nearest" +
          " neighbors you would like to find. Your options are:")
    for idx in range(len(all_titles)):
        print("\t" + str(idx) + ". " + all_titles[idx])

    # Prompt the user to choose an article
    while True:
        choice = input("Enter your choice here: ")
        try:
            choice = int(choice)
            break
        except ValueError:
            print("Error: you must enter an integer between 0 and " +
                  str(len(all_titles) - 1) + ", inclusive.")

    # Compute and print the results
    neighbours = nearest_neighbors(mat, all_titles, all_titles[choice], 5)
    print("\nThe 5 nearest neighbors of " + all_titles[choice] + " are:")
    for nbr in neighbours:
        print("\t" + nbr)


# Leave the following line commented when you submit your code to OwlTest/CanvasTest,
# but uncomment it to perform the analysis for the discussion questions.
run()

# if __name__ == '__main__':
    # x, y = get_title_and_text('wikipedia_articles/rice_university.xml')
    # words = get_words(y)
    # words_dict = count_words(words)[1]
    # words_dict = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse=True))
    # for k, v in words_dict.items():
    #     print(f"{k}:{v}")

    # x, y = get_title_and_text('wikipedia_articles/abolitionism.xml')
    # words = get_words(y)
    # words_dict = count_words(words)[1]
    # words_dict = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse=True))
    # for k, v in words_dict.items():
    #     print(f"{k}:{v}")
