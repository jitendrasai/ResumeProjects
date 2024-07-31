#Jitendra Sai

from tkinter import *
from tkinter import filedialog, ttk, messagebox
from tktooltip import ToolTip
import sqlite3
import io
import time
import datetime
import glob, os
from PIL import Image, ImageTk, ImageOps
from pathlib import Path
import numpy as np
import cv2
import math

# Facebook GraphAPI
import facebook
import requests
import json
import tablib

# This is for data processing
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
# Tag the words. pos = parts of speech
from nltk import pos_tag
# Removing punctuations https://www.youtube.com/watch?v=U8m5ug9Q54M
import re

# Stopword removal
nltk.download('stopwords')
from nltk.corpus import stopwords
# Stemming
from nltk.stem.porter import PorterStemmer
# Lemmatizatoin
from nltk.stem.wordnet import WordNetLemmatizer
from resnet.query_using_resnet_features import Resnet
from resnet.index_using_resnet_pretrained import feature_extract_resnet

# Header for the excel
headers = (
'id', 'name', 'created_time', 'link', 'file_name', 'full_picture', 'place', 'message', 'message_tags', 'description',
'comments', 'caption', 'reactions', 'story_tags', 'type', 'status_type', 'from', 'subscribed')
dlist = [('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17')]


def get_data(feed):
    if feed.get('type') is not None:
        if feed['type'] == 'photo':
            if feed.get('message') is not None:

                if feed.get('id') is not None:
                    id = feed['id']
                else:
                    id = ' NA '

                if feed.get('name') is not None:
                    name = feed['name']
                else:
                    name = 'NA'

                if feed.get('created_time') is not None:
                    createdtime = feed['created_time']
                else:
                    createdtime = ' NA '

                if feed.get('link') is not None:
                    link = feed['link']
                else:
                    link = ' NA '

                if feed.get('full_picture') is not None:
                    print("Start Download")
                    full_picture = feed['full_picture']
                    # tokenize and get image name from url between / and ?
                    r1 = full_picture.partition("?")[0]
                    r2 = r1.split("/")
                    filename = r2.pop()
                    # print('getdata:'+ filename)
                    r = requests.get(full_picture, allow_redirects=True)
                    open(filename, 'wb').write(r.content)
                    print("Stop download")
                else:
                    full_picture = ' NA '

                if feed.get('place') is not None:
                    place = feed['place']
                else:
                    place = ' NA '

                if feed.get('message') is not None:
                    message = feed['message']
                else:
                    message = ' NA '

                if feed.get('message_tags') is not None:
                    message_tags = feed['message_tags']
                else:
                    message_tags = ' NA '

                if feed.get('description') is not None:
                    description = feed['description']
                else:
                    description = ' NA '

                if feed.get('comments') is not None:
                    comments = feed['comments']
                else:
                    comments = 'NA'

                if feed.get('caption') is not None:
                    caption = feed['caption']
                else:
                    caption = ' NA '

                if feed.get('reactions') is not None:
                    reactions = feed['reactions']
                else:
                    reactions = ' NA '

                if feed.get('story_tags') is not None:
                    story_tags = feed['story_tags']
                else:
                    story_tags = ' NA '

                if feed.get('type') is not None:
                    type = feed['type']
                else:
                    type = ' NA '

                if feed.get('status_type') is not None:
                    status_type = feed['status_type']
                else:
                    status_type = ' NA '

                if feed.get('from') is not None:
                    fromk = feed['from']
                else:
                    fromk = ' NA '

                if feed.get('subscribed') is not None:
                    subscribed = feed['subscribed']
                else:
                    subscribed = ' NA '

                dlist.append((id, name, createdtime, link, filename, full_picture, place, message, message_tags,
                              description, comments, caption, reactions, story_tags, type, status_type, fromk,
                              subscribed))


# create directory
def createDirectory(path):
    # Create output directory
    output_dir = path + '\\metadata'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Change directory path to output directory
    os.chdir(output_dir)


# Clean, tokenize, classify
def processData():
    # Read extracted data, clean, tokenize and classify
    data = pd.read_excel(r'userdata.xlsx')
    df = pd.DataFrame(data, columns=['id', 'created_time', 'file_name', 'link', 'place', 'message'])
    # print(df)

    all_verbs = []
    all_adj = []
    all_nouns = []

    for i in df.index:
        first_dialogue = df.loc[i, "message"]
        # print(first_dialogue)
        # lower case
        clean_text_1 = []
        clean_text_1.append(str(first_dialogue).lower())
        # word tokenize
        clean_text_2 = []
        nltk.download('punkt')
        clean_text_2 = [word_tokenize(i) for i in clean_text_1]
        # Removing punctuations https://www.youtube.com/watch?v=U8m5ug9Q54M
        clean_text_3 = []
        for words in clean_text_2:
            for w in words:
                res = re.sub(r'[^\w\s]', "", w)
                if res != "":
                    clean_text_3.append(res)
        # Stopword removal
        clean_text_4 = []
        for word in clean_text_3:
            if not word in stopwords.words('english'):
                clean_text_4.append(word)
        # Stemming
        port = PorterStemmer()
        clean_text_5 = [port.stem(i) for i in clean_text_4]
        # Lemmatizatoin
        wnet = WordNetLemmatizer()
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        clean_text_6 = []
        for words in clean_text_4:
            clean_text_6.append(wnet.lemmatize(words))
        tagged_tokens = nltk.pos_tag(clean_text_6)

        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        # CD: numeral, cardinal
        # IN: preposition or conjunction, subordinating
        # JJ: adjective or numeral, ordinal
        # MD: modal auxiliary
        # NN: noun, common, singular or mass
        # NNS: noun, common, plural
        # RB: adverb
        # VB: verb, base form
        # VBD: verb, past tense
        # VBG: verb, present participle or gerund
        # VBZ: verb, present tense, 3rd person singular

        verbs = [token[0] for token in tagged_tokens if token[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        adj = [token[0] for token in tagged_tokens if token[1] in ['JJ', 'JJR', 'JJS']]
        nouns = [token[0] for token in tagged_tokens if token[1] in ['NN']]
        # print("Verbs: " , verbs)
        # print("Adjectives: ", adj)
        # print("Nouns: ", nouns)
        # print("\n\n")
        all_verbs.append(verbs)
        all_adj.append(adj)
        all_nouns.append(nouns)

    # To save it back as Excel
    df['Verbs'] = all_verbs
    df['Adj'] = all_adj
    df['Nouns'] = all_nouns
    df.to_excel(r'NLP-userdata.xlsx')


# get feed
def getFeeds(access_token, path):
    graph = facebook.GraphAPI(access_token)
    user = graph.get_object("me")

    # Fetch required feed data using an API
    feeds = graph.get_connections(user["id"], "feed",
                                  fields='id,created_time,name,message,link,file_name,full_picture,subscribed,caption,comments,description,from,message_tags,place,reactions,story_tags,type,status_type')

    # Create a folder and save downloaded images and metadata
    createDirectory(path)

    # while loop to keep paginating requests until finished.
    while True:
        try:
            # Get feed data and filter required fields.
            [get_data(feed=feed) for feed in feeds["data"]]
            # a request to the next page of data, if it exists.
            feeds = requests.get(feeds["paging"]["next"]).json()
        except KeyError:
            # When there are no more feeds (['paging']['next']), break from the
            # loop and end the script.
            break

    # Extract data into excel
    ds2 = tablib.Dataset(*dlist, headers=headers)
    with open('userdata.xlsx', 'wb') as f1:
        f1.write(ds2.export("xlsx"))

    processData()
    download_label["text"] = "Download Done."


# Store access token
def loginWithAccesstoken(access_token):
    try:
        graph = facebook.GraphAPI(access_token)
        user = graph.get_object("me")
        login_label["text"] = "Login Successful."
        download_browse.config(state='normal')
    except Exception as e:
        login_label["text"] = e


class InnerCanvas:
    # Class used for smaller, 3x3 canvas display in right panel

    def __init__(self, filepath):
        # Variables: canvas, filepath, image_display, selected
        # canvas - tkinter Canvas object

        # filepath - string, should be full file name and path

        # image_display - reference to the image itself, via Pillow's ImageTK.
        # Necessary. If not there, image will not display outside of function

        # selected - boolean that determines whether the user has selected the image or not

        self.canvas = Canvas(dir_canvas, width=95, height=95)
        self.filepath = filepath

        self.selected = True

        temp_img = Image.open(self.filepath).convert('RGBA')
        temp_img = ImageOps.exif_transpose(temp_img)
        temp_img = temp_img.resize((110, 110))  # process image for correct display size

        self.image_display = ImageTk.PhotoImage(temp_img)  # the image itself- MUST save this for display to work!
        self.canvas.create_image(0, 0, anchor=NW, image=self.image_display)
        self.canvas.update()


class InnerCanvas1:
    # Class used for smaller, 3x3 canvas display in right panel

    def __init__(self, filepath):
        self.canvas = Canvas(fb_canvas, width=95, height=95)
        self.filepath = filepath

        self.selected = True

        temp_img = Image.open(self.filepath).convert('RGBA')
        temp_img = ImageOps.exif_transpose(temp_img)
        temp_img = temp_img.resize((110, 110))  # process image for correct display size

        self.image_display = ImageTk.PhotoImage(temp_img)  # the image itself- MUST save this for display to work!
        self.canvas.create_image(0, 0, anchor=NW, image=self.image_display)
        self.canvas.update()

    # Database adapter and converter functions


# Adapt: Accepts histogram as numpy array as parameter, saves it to binary file and converts it to raw binary (blob)
# Convert: Accepts binary blob and converts back into numpy array
def adapt(np_array):
    outfile = io.BytesIO()
    np.save(outfile, np_array)
    outfile.seek(0)
    return sqlite3.Binary(outfile.read())


def convert(blob):
    return np.load(io.BytesIO(blob))


# Prep Database: Establishes adapter/converter functions and connection to database
# Establishes file named LocalPics.db and initial table
def prep_database():
    global connection

    sqlite3.register_adapter(np.ndarray, adapt)
    sqlite3.register_converter("NPARRAY", convert)

    connection = sqlite3.connect('LocalPics.db', detect_types=sqlite3.PARSE_DECLTYPES)

    connection.execute('''CREATE TABLE if NOT EXISTS LocalPics
        (Filename TEXT NOT NULL,
        EpochTime FLOAT,
        Histogram NPARRAY,
        Annotation TEXT);''')


# DB Entry: Accepts filename (full file name + path) and numpy array histogram as parameters.
# Performs insert query and commits to DB
def db_entry(filename, hist):
    connection.execute("INSERT INTO LocalPics (Filename, EpochTime, Histogram) \
      VALUES ('" + str(filename) + "', " + str(os.path.getmtime(filename)) + ", (?))", (hist,));
    connection.commit()


# Create DB: Accepts folder path (base image directory) as parameter
# Calls prep_database, then goes through each image in folder
# Will either load from DB or generate new histogram and enter into DB depending on whether image was found or not
def create_db(folder, flag):
    start = time.time()

    prep_database()

    if not os.path.isdir(folder) or not os.listdir(folder):  # if folder paramater is not a directory, or is empty
        dir_label.config(text="Error: Path is not a directory or is empty")
        root.update()
        return

    i = 0
    for file in os.listdir(folder):

        if (file.lower().endswith(('.png', '.jpg', '.jpeg'))):
            # print('Create DB A:'+file)
            name = folder + "/" + file
            f = Path(name)  # get Path variable for file
            # print('Create DB B:' + str(f))
            cursor = connection.execute("SELECT Filename FROM LocalPics WHERE Filename = ?", (str(f),))

            if (
                    cursor.fetchone() == None):  # if no matching item was found, read image, get histogram, and store in database
                try:  # histogram generation
                    img = cv2.imread(name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    db_entry(f, hist)
                except:  # try block won't work if file is not valid image file
                    dir_label.config(
                        text="ERROR - Directory does not\n contain only image files.\n Please reload DB and try again")
                    return

            names.append(str(f))  # stores f in names list
            i = i + 1
            dir_label.config(text=(str(i) + "/" + str(
                len(os.listdir(folder))) + " Images Loaded..."))  # assumes folder is all images
            root.update()

    cursor = connection.execute("SELECT DISTINCT Annotation FROM LocalPics")  # obtain annotations from database

    ann_list.clear()  # clear list of previously loaded annotations, if any

    for row in cursor:
        # print(row)
        ann_list.append(row[0])  # add annotation to array

    ann_select.config(values=ann_list, state='readonly')
    ann_select.current(0)

    dir_label.config(text="All " + str(i) + " images loaded")
    valid_dir.set(True)
    is_compare_valid()

    end = time.time()

    # print(end-start)

    clear_db_button.config(state='normal')
    # clear_local_db_button.config(state = 'normal')
    search_ann.config(state='normal')
    # If create/load db for facebook directory then enable load suggested tag button
    load_tag_button.config(state='normal')
    if (flag == 0):
        feature_extract_resnet(folder)
        compare_w_fb_button.config(state='normal')
        hist_slider.config(state='normal')
    root.update()


# Clear DB: Drops table, closes connection
def clear_db():
    ask_clear = messagebox.askquestion('Clear DB?',
                                       'Are you sure you want to clear the database?\n You will lose your annotations and need to recreate the database again!')
    if ask_clear == 'yes':
        connection.execute('DROP TABLE IF EXISTS LocalPics')
        dir_label.config(text="DB cleared. Please create a new \nDB to continue.")
        # Reset everything on UI
        dir_canvas.delete("all")
        annotate_label.config(text="")
        annotation_box.delete(0, END)
        tags_list = ['None']
        tags_select.config(values=tags_list, state='disabled')
        tags_select.current(0)
        ann_list = ['No Database Loaded']
        ann_select.config(values=ann_list, state='disabled')
        ann_select.current(0)
        set_boxes('disabled')
        ann_select.config(state='disabled')
        connection.commit()
        connection.close()


# Set Boxes: Lots of "state = normal" or "state = disabled" calls are made, this function simplifies some of them
def set_boxes(string):
    annotate_button.config(state=string)
    tag_button.config(state=string)
    annotation_box.config(state=string)
    search_ann.config(state=string)
    remove_button.config(state=string)
    load_tag_button.config(state=string)
    compare_w_fb_button.config(state=string)
    hist_slider.config(state=string)

    # Browse file: Opens file dialog window and displays whatever image the user


def browse_file(frame, button_title, canvas):
    fb_entry.config(state='normal')
    fb_entry.delete(0, END)

    if (download_opened.get() == True):
        directory = pathname
    else:
        directory = str(Path(Path.home())) + '/Pictures/'

    tempdir = filedialog.askopenfilename(parent=frame, initialdir=directory, title=(button_title))

    fb_entry.insert(END, tempdir)
    fb_entry.config(state='readonly')

    if tempdir:
        try:
            update_canvas_generic(tempdir, canvas, "fb")  # update canvas function- actually displays picture
            valid_fb.set(True)
        except:
            # print("invalid facebook image?")
            valid_fb.set(False)
    is_compare_valid()

    # Download browse function: Browse folder to download facebook images


def download_browse_func():
    download_button.config(state='normal')
    download_entry.config(state='normal')
    download_entry.delete(0, END)

    currdir = str(Path(Path.home())) + '/metadata/'
    tempdir = filedialog.askdirectory(parent=left_frame, initialdir=currdir, title=('Save Facebook Metadata'))

    download_entry.insert(END, tempdir)
    download_entry.config(state='readonly')

    # Browse directory: Very similar to above function, opens folder dialog, but also checks if directory is valid


def browse_dir(frame, button_title, fb_canvas):
    dir_label.config(text="")

    clear_db_button.config(state='disabled')

    load_db_button.config(state='disabled')
    load_tag_label.config(text="")

    back_button_fb.pack_forget()
    page_label_fb.pack_forget()
    next_button_fb.pack_forget()  # hides labels that encourages users to browse through pages of images

    global full_paths_fb_images
    full_paths_fb_images.clear()
    fb_subcanvases.clear()

    if directory_entry.get():
        old_dir = directory_entry.get()
    else:
        old_dir = "n/a"

    directory_entry.config(state='normal')
    directory_entry.delete(0, END)

    currdir = str(Path(Path.home())) + '/Pictures/'
    tempdir = filedialog.askdirectory(parent=frame, initialdir=currdir, title=(button_title))

    if not tempdir or (opened_dir_once.get() and old_dir != tempdir):
        valid_dir.set(False)
        is_compare_valid()
    if tempdir:
        directory_entry.insert(END, tempdir)
        load_db_button.config(state='normal')
        opened_dir_once.set(True)

        if not os.path.isdir(tempdir) or not os.listdir(tempdir):  # if folder paramater is not a directory, or is empty
            dir_label.config(text="Error: Path is not a directory or is empty")
            root.update()
            return

        for file in os.listdir(tempdir):

            if (file.lower().endswith(('.png', '.jpg', '.jpeg'))):
                # print(file)
                name = tempdir + "/" + file
                f = Path(name)  # get Path variable for file
                # print(str(f))
                full_paths_fb_images.append(f)

        if len(full_paths_fb_images) != 0:
            # zipped = zip(res, full_paths_fb_images)
            # zipped = reversed(sorted(zipped))
            # res, full_paths_fb_images = map(list, zip(*zipped))

            back_button_fb.pack(side=LEFT, padx=(0, 10))
            page_label_fb.config(text="")
            page_label_fb.pack(side=LEFT)
            next_button_fb.pack(side=LEFT, padx=(10, 0))

            update_fb_canvas(full_paths_fb_images, fb_canvas, 0)
        else:
            fb_canvas.delete("all")
            load_db_button.config(state='disabled')
            page_label_fb.config(text="Check!! If selected folder has facebook images.")
            page_label_fb.pack(side=LEFT)

    directory_entry.config(state='readonly')


def browse_local_dir(frame, button_title):
    dir_label.config(text="")

    load_local_db_button.config(state='disabled')

    if local_directory_entry.get():
        old_dir = local_directory_entry.get()
    else:
        old_dir = "n/a"

    local_directory_entry.config(state='normal')
    local_directory_entry.delete(0, END)

    currdir = str(Path(Path.home())) + '/Pictures/'
    tempdir = filedialog.askdirectory(parent=frame, initialdir=currdir, title=(button_title))

    if not tempdir or (opened_dir_once.get() and old_dir != tempdir):
        valid_dir.set(False)
        is_compare_valid()
    if tempdir:
        local_directory_entry.insert(END, tempdir)
        load_local_db_button.config(state='normal')
        opened_dir_once.set(True)
    local_directory_entry.config(state='readonly')

    # Is compare valid: If the selected image is valid and directory is valid, enable comparison option


def is_compare_valid():
    if valid_fb.get() == True and valid_dir.get() == True:
        load_tag_button.config(state='normal')
        compare_w_fb_button.config(state='normal')
        hist_slider.config(state='normal')
    else:
        load_tag_button.config(state='disabled')
        compare_w_fb_button.config(state='disabled')
        hist_slider.config(state='disabled')

    # Update canvas: Displays resized version of selected facebook image


def update_canvas_generic(str, canvas, option):
    global fb_display

    temp_img = Image.open(str).convert('RGBA')
    temp_img = ImageOps.exif_transpose(temp_img)

    x_factor = temp_img.width / 300
    y_factor = temp_img.height / 300

    x = temp_img.width / x_factor
    y = temp_img.height / y_factor

    temp_img = temp_img.resize((int(x), int(y)))

    fb_display = ImageTk.PhotoImage(temp_img)
    canvas.create_image(0, 0, anchor=NW, image=fb_display)

    canvas.config(width=x, height=y)
    canvas.pack()
    canvas.update()


# check is list is empty
def isListEmpty(lis1):
    if '' in lis1:
        return 0
    else:
        return 1


# Load the tags for selected image from NLP analysed output
def load_tags(folder, fb_canvas):
    flag_selected = 0
    # Selected image file path
    image_file_path = ''

    for x in fb_subcanvases:
        if x.selected == True:
            # print("Filename Load tags A: " + str(x.filepath))
            image_file_path = str(x.filepath)
            flag_selected = 1

    if (flag_selected == 0):
        load_tag_label.config(text="Select image to load tags")
        tag_button.config(state='disabled')
        tags_list = ['None']
        tags_select.config(values=tags_list, state='disabled')
        tags_select.current(0)
        # print('else return')
        return

    verbs = []
    adjectives = []
    nouns = []
    tags_list = []
    flag = 0
    # Get selected image name from image file path
    split_filepath = image_file_path.split("\\")
    imagename = split_filepath.pop()
    # print(imagename)
    # Search for 'NLP-userdata.xlsx' in selected folder
    for file in os.listdir(folder):
        if (file.lower().endswith(('.xlsx'))):
            if (str(file) == 'NLP-userdata.xlsx'):
                # Search imagename in Facebook downloaded image data (NLP analysed data)
                userdata_filepath = folder + '/' + file
                # print(userdata_filepath)
                userdata = pd.read_excel(userdata_filepath)
                df = pd.DataFrame(userdata, columns=['file_name', 'Verbs', 'Adj', 'Nouns'])

                # compare slected image name
                for i in df.index:
                    file_name = df.loc[i, "file_name"]
                    # print('filename load tags B:'+file_name)

                    if (str(file_name) == str(imagename)):
                        ##print('True')
                        # get the Verbs,Adjectives,Nouns
                        verbs = df.loc[i, "Verbs"].strip("[]").replace("'", "").split(", ")
                        adjectives = df.loc[i, "Adj"].strip("[]").replace("'", "").split(", ")
                        nouns = df.loc[i, "Nouns"].strip("[]").replace("'", "").split(", ")
                        # print(verbs)
                        # print(adjectives)
                        # print(nouns)

                        if isListEmpty(verbs):
                            tags_list.extend(verbs)
                        if isListEmpty(adjectives):
                            tags_list.extend(adjectives)
                        if isListEmpty(nouns):
                            tags_list.extend(nouns)
                        # print(tags_list)
                        flag = 1
                        tags_select.config(values=tags_list, state='readonly')
                        tags_select.current(0)
                        tag_button.config(state='normal')
                        load_tag_label.config(text="")

    if (flag == 0):
        tags_list = []
        fb_canvas.delete("all")
        load_tag_label.config(text="Select folder having \ndownloaded data")


# Histogram comparison: Accepts image_file (full file path + filename), folder, and canvas
def hist_comp(folder, dir_canvas):
    resnet_obj =  Resnet()
    back.pack_forget()
    page_label.pack_forget()
    next.pack_forget()  # hides labels that encourages users to browse through pages of images

    # Selected image file path
    image_file = ''

    for x in fb_subcanvases:
        if x.selected == True:
            # print("Filename hist_comp A:" + str(x.filepath))
            image_file = str(x.filepath)

    # print("Image File " + image_file)
    threshold = hist_slider_val.get() / 100
    full_paths=resnet_obj.query_image(image_file,threshold)
    subcanvases.clear()

    if len(full_paths) != 0:
        # zipped = zip(res, full_paths)
        # zipped = reversed(sorted(zipped))
        # res, full_paths = map(list, zip(*zipped))

        back.pack(side=LEFT, padx=(0, 10))
        page_label.config(text="")
        page_label.pack(side=LEFT)
        next.pack(side=LEFT, padx=(10, 0))

        update_canvas_w_subcanvas(full_paths, dir_canvas, 0)
    else:
        dir_canvas.delete("all")
        page_label.config(text="No valid matches found.\n Try lowering the threshold value.")
        page_label.pack(side=LEFT)


# Update canvas with subcanvas: Parameters are path_array (the array of valid results from histogram comparison function)
# canvas
def update_canvas_w_subcanvas(path_array, dir_canvas, integer):
    # timeframe_label.config(text="")
    back.config(state='disabled')
    next.config(state='disabled')

    old_count = count.get()
    count.set(integer)

    i = 50
    j = 50

    possible_max = count.get() + 9
    actual_max = min(possible_max, len(path_array))

    current_page = int((count.get() / 9) + 1)
    max_pages = int(math.ceil(len(path_array) / 9))

    dir_canvas.delete("all")
    page_label.config(text="Page " + str(current_page) + " of " + str(max_pages))

    # NOTE: the sizing and spacing of the display function is hard-coded (ex. val + 100, val + 50)

    # if the length of the subcanvas array is greater than what the max is, then the values have already been generated
    index = 0;
    if (len(subcanvases) >= actual_max):
        # print("A - display")
        for x in range(count.get(), count.get() + 9):
            if (x < len(path_array)):
                dir_canvas.create_window(i, j, window=subcanvases[count.get() + index].canvas)
                i = i + 100
                if (((index + 1) % 3) == 0):
                    i = 50
                    j = j + 100

                if subcanvases[index].selected == True:
                    subcanvases[index].canvas.config(highlightbackground='red', highlightthickness=2)

                subcanvases[index].canvas.bind("<Button-1>", lambda_helper(subcanvases[index]))
                subcanvases[index].canvas.bind("<Double-Button-1>", lambda_helper2(subcanvases[index]))
                index = index + 1

    else:  # else, we need to generate some subcanvases
        # print("B - generate")
        index = count.get()
        for x in path_array[count.get():actual_max]:
            ic = InnerCanvas(x)
            subcanvases.append(ic)
            dir_canvas.create_window(i, j, window=subcanvases[index].canvas)
            i = i + 100
            if (((index + 1) % 3) == 0):
                i = 50
                j = j + 100
            if subcanvases[index].selected == True:
                subcanvases[index].canvas.config(highlightbackground='red', highlightthickness=2)
            subcanvases[index].canvas.bind("<Button-1>", lambda_helper(subcanvases[index]))
            subcanvases[index].canvas.bind("<Double-Button-1>", lambda_helper2(subcanvases[index]))
            index = index + 1

    # print("subcanvases now at size ", len(subcanvases))

    dir_canvas.pack()

    # which buttons of back/next should be accessible?
    if current_page == 1 and current_page != max_pages:
        next.config(state='normal')
    elif current_page == max_pages and max_pages != 1:
        back.config(state='normal')
    elif current_page > 1 and current_page < max_pages:
        back.config(state='normal')
        next.config(state='normal')

    annotation_box.config(state='normal')
    set_boxes('normal')
    dir_canvas.update()


# Update canvas with subcanvas: Parameters are path_array (the array of valid results from histogram comparison function)
# canvas
def update_fb_canvas(path_array, fb_canvas, integer):
    back_button_fb.config(state='disabled')
    next_button_fb.config(state='disabled')

    # print(len(path_array))

    old_count = count.get()
    count.set(integer)

    i = 50
    j = 50

    possible_max = count.get() + 9
    actual_max = min(possible_max, len(path_array))

    current_page = int((count.get() / 9) + 1)
    max_pages = int(math.ceil(len(path_array) / 9))

    fb_canvas.delete("all")
    page_label_fb.config(text="Page " + str(current_page) + " of " + str(max_pages))

    # NOTE: the sizing and spacing of the display function is hard-coded (ex. val + 100, val + 50)

    # if the length of the subcanvas array is greater than what the max is, then the values have already been generated
    index = 0;
    if (len(fb_subcanvases) >= actual_max):
        # print("A - display")
        for x in range(count.get(), count.get() + 9):
            if (x < len(path_array)):
                fb_canvas.create_window(i, j, window=fb_subcanvases[count.get() + index].canvas)
                i = i + 100
                if (((index + 1) % 3) == 0):
                    i = 50
                    j = j + 100

                fb_subcanvases[index].selected = False

                if fb_subcanvases[index].selected == True:
                    fb_subcanvases[index].canvas.config(highlightbackground='red', highlightthickness=2)

                fb_subcanvases[index].canvas.bind("<Button-1>", lambda_helper(fb_subcanvases[index]))
                fb_subcanvases[index].canvas.bind("<Double-Button-1>", lambda_helper2(fb_subcanvases[index]))
                index = index + 1

    else:  # else, we need to generate some fb_subcanvases
        # print("B - generate")
        index = count.get()
        for x in path_array[count.get():actual_max]:
            ic = InnerCanvas1(x)
            fb_subcanvases.append(ic)
            fb_canvas.create_window(i, j, window=fb_subcanvases[index].canvas)
            i = i + 100
            if (((index + 1) % 3) == 0):
                i = 50
                j = j + 100

            fb_subcanvases[index].selected = False

            if fb_subcanvases[index].selected == True:
                fb_subcanvases[index].canvas.config(highlightbackground='red', highlightthickness=2)
            fb_subcanvases[index].canvas.bind("<Button-1>", lambda_helper(fb_subcanvases[index]))
            fb_subcanvases[index].canvas.bind("<Double-Button-1>", lambda_helper2(fb_subcanvases[index]))
            index = index + 1

    # print("fb_subcanvases now at size ", len(fb_subcanvases))

    fb_canvas.pack()

    # which buttons of back/next should be accessible?
    if current_page == 1 and current_page != max_pages:
        next_button_fb.config(state='normal')
    elif current_page == max_pages and max_pages != 1:
        back_button_fb.config(state='normal')
    elif current_page > 1 and current_page < max_pages:
        back_button_fb.config(state='normal')
        next_button_fb.config(state='normal')

    set_boxes('normal')
    fb_canvas.update()


# Lambda helpers: Establishes separate events for each click event assigned in a loop (for above function)
# Without these, only last canvas displayed will have click events tied to it
def lambda_helper(obj):
    return lambda event: subcanv_func(event, obj)


def lambda_helper2(obj):
    return lambda event: open_img(event, obj)


# Subcanvas function: Sets selected value for InnerCanvas object, sets border as necessary
def subcanv_func(event, ic):
    if (ic.selected == True):
        ic.canvas.config(highlightthickness=0)
        ic.selected = False
    else:
        ic.canvas.config(highlightbackground='red', highlightthickness=2)
        ic.selected = True


def open_img(event, ic):
    os.startfile(ic.filepath)

    # Get Val: Function that obtains the value of whatever the menu slider is set to and displays it


def get_val(settings_modified, hist_slider_label):
    new_val = float(hist_slider_val.get()) / 100.0
    hist_slider_label.config(text=new_val)


def open_help(root):
    # cwd = os.getcwd()
    global cwd
    path = str(cwd + '//' + 'ReadMe.htm')
    os.system(path)


# Annotation search: Finds images with a certain annotation and displays in right panel
def annotation_search(dir_canvas, string):
    # print('String : ' + string)
    back.pack_forget()
    page_label.pack_forget()
    next.pack_forget()
    full_paths.clear()

    subcanvases.clear()

    if (string == "None"):
        query = "SELECT DISTINCT Filename FROM LocalPics WHERE Annotation IS NULL"
    else:
        query = "SELECT DISTINCT Filename FROM LocalPics WHERE Annotation = '" + str(string) + "'"

    for row in connection.execute(query):
        full_paths.append(row[0])  # add successful search into array

    back.pack(side=LEFT, padx=(0, 10))
    page_label.config(text="")
    page_label.pack(side=LEFT)
    next.pack(side=LEFT, padx=(10, 0))

    subcanvases.clear()
    update_canvas_w_subcanvas(full_paths, dir_canvas, 0)  # upadte display


# Remove annotation function: removes annotation for any selected subcanvas.
# Updates list of annotations afterwards
def remove_annotation():
    count = 0
    for x in subcanvases:
        if x.selected == True:
            connection.execute("DELETE FROM LocalPics WHERE Filename = ? AND Annotation = ?",
                               (str(x.filepath), str(annotation_box.get()),))
            count = count + 1

    connection.commit()
    annotate_label.config(text="Removed annotations " + str(annotation_box.get()) + " from " + str(count) + " images")

    cursor = connection.execute("SELECT DISTINCT Annotation FROM LocalPics")

    ann_list.clear()  # clear list of previously loaded annotations, if any
    for row in cursor:
        ann_list.append(row[0])  # add annotation to array

    ann_select.config(values=ann_list, state='readonly')
    annotation_box.delete(0, END)


# Annotate image function
# Determines which subcanvas images are selected, updates canvas with annotation specified by user.
# Assumes annotation box is not empty
def annotate_img():
    count = 0

    for x in subcanvases:
        if x.selected == True:
            # print("Filename Annotate Img A:" + str(x.filepath))
            connection.execute("INSERT INTO LocalPics (Filename, Annotation) \
      VALUES ('" + str(x.filepath) + "', (?))", (str(annotation_box.get()),));
            count = count + 1

    connection.commit()
    annotate_label.config(text="Added annotation '" + str(annotation_box.get()) + "' to " + str(count) + " images")

    cursor = connection.execute("SELECT DISTINCT Annotation FROM LocalPics")

    ann_list.clear()  # clear list of previously loaded annotations, if any
    for row in cursor:
        ann_list.append(row[0])  # add annotation to array

    ann_select.config(values=ann_list, state='readonly')


# Annotate images from suggested tags
def tag_img():
    count = 0

    # annotate images selected from facebook directory subcanvas
    for x in fb_subcanvases:
        if x.selected == True:
            # print("A Filename" + str(x.filepath))
            # print(str(annotation_box.get()))
            connection.execute("INSERT INTO LocalPics (Filename, Annotation) \
      VALUES ('" + str(x.filepath) + "', (?))", (str(tags_select.get()),));
            count = count + 1

    # annotate images selected from local directory subcanvas
    for x in subcanvases:
        if x.selected == True:
            x.filepath = x.filepath.decode('utf-8')
            # print("B Filename" + str(x.filepath))
            connection.execute("INSERT INTO LocalPics (Filename, Annotation) \
      VALUES ('" + str(x.filepath) + "', (?))", (str(tags_select.get()),));
            count = count + 1

    connection.commit()
    annotate_label.config(text="Added annotation '" + str(tags_select.get()) + "' to " + str(count) + " images")

    cursor = connection.execute("SELECT DISTINCT Annotation FROM LocalPics")

    ann_list.clear()  # clear list of previously loaded annotations, if any
    for row in cursor:
        ann_list.append(row[0])  # add annotation to array

    ann_select.config(values=ann_list, state='readonly')


### VISUAL COMPONENTS ###

# VARIABLES
subcanvases = []
fb_subcanvases = []
res = []
full_paths = []
full_paths_fb_images = []
names = []
bg_color = '#e2e3e7'
global cwd
cwd = os.getcwd()

# VISUALS CONFIG
root = Tk()
root.title("Personalised Image Annotations Using Facebook")
root.geometry('1000x650')
root.resizable(0, 0)

#  TKINTER VARS
count = IntVar()

opened_dir_once = BooleanVar()
settings_modified = BooleanVar()
download_opened = BooleanVar()
hist_slider_val = DoubleVar()
hist_slider_val.set(60)

valid_fb = BooleanVar()
valid_dir = BooleanVar()
valid_local_dir = BooleanVar()

# WINDOW/FRAME SETUP
window = PanedWindow(root, orient=HORIZONTAL)
window.pack(fill=BOTH, expand=True)

right_frame = Frame(window, bg=bg_color)
mid_frame = Frame(window, bg=bg_color)
left_frame = Frame(window, bg=bg_color)

window.add(left_frame)
window.add(mid_frame)
window.add(right_frame)

# ORGANIZE FRAMES

right_frame.grid(row=0, column=2)
mid_frame.grid(row=0, column=1)
left_frame.grid(row=0, column=0)

window.paneconfig(left_frame, width=250, height=600)
window.paneconfig(mid_frame, width=400, height=600)
window.paneconfig(right_frame, width=350, height=600)

# START LEFT PANEL

top_left_frame = Frame(left_frame, bg=bg_color, width=250, height=300)
mid_left_frame = Frame(left_frame, bg=bg_color, width=250, height=100)
bottom_left_frame = Frame(left_frame, bg=bg_color, width=250, height=200)

top_left_frame.pack()
mid_left_frame.pack()
bottom_left_frame.pack()

Label(top_left_frame, text="Facebook Login", font=20, bg=bg_color).pack(pady=(10, 0))

(Label(top_left_frame, text="Access Token", bg=bg_color)).pack(pady=(10, 2))
accesstoken = Entry(top_left_frame, textvariable="accesstoken")
accesstoken.pack()

login_button = Button(top_left_frame, text="Login", width=10, height=1,
                      command=lambda: loginWithAccesstoken(accesstoken.get()))
login_button.pack(pady=(8, 0))

login_label = Label(top_left_frame, text="", bg=bg_color)
login_label.pack(pady=(5, 0))

(Label(mid_left_frame, text="Download contents to: ", bg=bg_color)).pack(pady=(60, 0))

download_entry = Entry(mid_left_frame, width=25, state="readonly", textvariable="download")
download_entry.pack(side=LEFT, padx=(0, 5), pady=(5, 0))

download_browse = Button(mid_left_frame, text="Browse", state='disabled', width=10, height=1,
                         command=lambda: download_browse_func())
download_browse.pack(side=RIGHT, pady=(5, 0))

download_button = Button(bottom_left_frame, text="Download", state='disabled', width=10, height=1,
                         command=lambda: getFeeds(accesstoken.get(), download_entry.get()))
download_button.pack(pady=(8, 0))

download_label = Label(bottom_left_frame, text=" ", bg=bg_color)
download_label.pack(pady=(5, 0))

help_button = Button(bottom_left_frame, text="Help", width=10, height=1, command=lambda: open_help(root))
help_button.pack(side=BOTTOM, pady=(180, 0))

# END LEFT PANEL

# START MID PANEL

new_top_frame = Frame(mid_frame, bg=bg_color, width=400, height=300)
top_mid_frame = Frame(mid_frame, bg=bg_color, width=400, height=50)
canvas_mid_frame = Frame(mid_frame, bg=bg_color, width=400, height=300)
bottom_mid_frame = Frame(mid_frame, bg=bg_color, width=400, height=50)

top_mid_frame.pack()
bottom_mid_frame.pack()
canvas_mid_frame.pack()
new_top_frame.pack()

Label(top_mid_frame, font=20, text="FB Directory Selection", bg=bg_color).grid(row=0, pady=(10, 0), padx=(85, 0))

# Browse facebook metadata directory
directory_entry = Entry(top_mid_frame, width=50, state="readonly")
directory_button = Button(top_mid_frame, text="Browse", width=10, height=1,
                          command=lambda: browse_dir(top_mid_frame, 'Facebook Metadata Directory', fb_canvas))
directory_label = Label(top_mid_frame, text="")

page_label_fb = Label(bottom_mid_frame, bg=bg_color,
                      text="Please select personal facebook directory.\n", font=('Arial', 8))
page_label_fb.pack()

back_button_fb = Button(bottom_mid_frame, text='Back',
                        command=lambda: update_fb_canvas(full_paths_fb_images, fb_canvas, int(count.get() - 9)))
next_button_fb = Button(bottom_mid_frame, text='Next',
                        command=lambda: update_fb_canvas(full_paths_fb_images, fb_canvas, int(count.get() + 9)))

fb_canvas = Canvas(canvas_mid_frame, width=300, height=296)
fb_canvas.pack()

load_db_button = Button(new_top_frame, text="Create/Load DB", state='disabled',
                        command=lambda: create_db(directory_entry.get(), 1))
dir_label = Label(new_top_frame, text="", bg=bg_color)
load_tag_button = Button(new_top_frame, text="Load Suggested Tags", state='disabled',
                         command=lambda: load_tags(directory_entry.get(), fb_canvas))
load_tag_label = Label(new_top_frame, text="", bg=bg_color)

clear_db_button = Button(new_top_frame, text="Clear DB", state='disabled', command=lambda: clear_db())

directory_entry.grid(row=2, column=0, padx=(5), pady=(5, 5))
directory_button.grid(row=2, column=1, pady=(5, 5))
load_db_button.grid(row=3, column=0, pady=(5, 0), padx=(0, 210))
dir_label.grid(row=3, column=0, pady=(0, 0), padx=(100, 0))
clear_db_button.grid(row=3, column=1, pady=(5, 0))
load_tag_button.grid(row=4, column=0, padx=(0, 178), pady=(10, 0))
load_tag_label.grid(row=4, column=0, pady=(10, 0), padx=(178, 0))

tags_list = ['None']
tags_select = ttk.Combobox(new_top_frame, values=tags_list, state='disabled')
tags_select.current(0)
tags_select.grid(row=5, column=0, pady=(10, 0), padx=(0, 160))

tag_button = Button(new_top_frame, text="Annotate", state='disabled', command=lambda: tag_img())
tag_button.grid(row=5, column=1, pady=(10, 0))

annotate_label = Label(new_top_frame, text="", bg=bg_color)
annotate_label.grid(columnspan=2, row=6, pady=(10, 0))

# END MID PANEL

# START RIGHT PANEL
top_right_frame = Frame(right_frame, bg=bg_color, width=350, height=100)
mid_right_frame = Frame(right_frame, bg=bg_color, width=350, height=100)
canvas_right_frame = Frame(right_frame, bg=bg_color, width=350, height=250)
bottom_right_frame = Frame(right_frame, bg=bg_color, width=350, height=250)

top_right_frame.pack()
mid_right_frame.pack()
canvas_right_frame.pack()
bottom_right_frame.pack()

Label(top_right_frame, font=20, text="Personal Directory Contents", bg=bg_color).grid(row=0, pady=(10, 0), padx=(10, 0))

# Browse local directory from system to search facebook image in local directory
local_directory_entry = Entry(top_right_frame, width=40, state="readonly")
local_directory_button = Button(top_right_frame, text="Browse", width=10, height=1,
                                command=lambda: browse_local_dir(top_right_frame, 'Local Directory'))
local_directory_label = Label(top_right_frame, text="")

load_local_db_button = Button(top_right_frame, text="Create/Load DB", state='disabled',
                              command=lambda: create_db(local_directory_entry.get(), 0))
compare_w_fb_button = Button(top_right_frame, text="Compare", state='disabled',
                             command=lambda: hist_comp(local_directory_entry.get(), dir_canvas))

Label(top_right_frame, text="Threshold", bg=bg_color).grid(row=3, column=0, padx=(0, 150), pady=(5, 5))
hist_slider_label = Label(top_right_frame)
hist_slider_label.config(text=float(hist_slider_val.get()) / 100.0)
hist_slider = Scale(top_right_frame, from_=0.0, to=100, tickinterval=0.1, orient='horizontal', showvalue=0,
                    variable=hist_slider_val, state='disabled',
                    command=lambda x: get_val(settings_modified, hist_slider_label))
ToolTip(hist_slider,
        msg="Threshold in which an image is considered a correct\n potential match when using the compare function.\n"
            + "Lower values return more results.")

local_directory_entry.grid(row=1, column=0, padx=(0, 5), pady=(5, 5))
local_directory_button.grid(row=1, column=1, padx=(0, 5), pady=(5, 5))
load_local_db_button.grid(row=2, column=0, pady=(5, 5), padx=(0, 150))
compare_w_fb_button.grid(row=2, column=1, pady=(5, 5))
hist_slider_label.grid(row=3, column=1, pady=(5, 5))
hist_slider.grid(row=3, column=0, padx=(100, 0), pady=(5, 5))

# Canvas image page
page_label = Label(mid_right_frame, bg=bg_color,
                   text="No directory images searches loaded.\n" +
                        "Search database by annotation.", font=('Arial', 8))
page_label.pack()

back = Button(mid_right_frame, text='Back',
              command=lambda: update_canvas_w_subcanvas(full_paths, dir_canvas, int(count.get() - 9)))
next = Button(mid_right_frame, text='Next',
              command=lambda: update_canvas_w_subcanvas(full_paths, dir_canvas, int(count.get() + 9)))

# canvas
dir_canvas = Canvas(canvas_right_frame, width=300, height=296)
dir_canvas.pack()

# Search by Annotation
ann_list = ['No Database Loaded']
ann_select = ttk.Combobox(bottom_right_frame, values=ann_list, state='disabled')
ann_select.current(0)
Label(bottom_right_frame, text="Search by annotation", bg=bg_color).grid(row=1, column=0, pady=(10, 50), padx=(35, 30))
ann_select.grid(row=1, column=0, pady=(10, 0), padx=(35, 30))

search_ann = Button(bottom_right_frame, text="Search by Annotation",
                    state='disabled', command=lambda: annotation_search(dir_canvas, ann_select.get()))
search_ann.grid(row=1, column=1, pady=(10, 0), padx=(0, 40))

Label(bottom_right_frame, bg=bg_color, text="Enter Annotation:").grid(row=2, column=0, pady=(0, 100), padx=(0, 30))

annotation_box = Entry(bottom_right_frame, textvariable="ann", state='disabled')
annotation_box.grid(row=2, column=1, pady=(0, 100), padx=(0, 30))

annotate_button = Button(bottom_right_frame, text="Annotate", state='disabled', command=lambda: annotate_img())
annotate_button.grid(row=2, column=0, pady=(0, 40), padx=(0, 30))

remove_button = Button(bottom_right_frame, text="Remove Annotation", state='disabled',
                       command=lambda: remove_annotation())
remove_button.grid(row=2, column=1, pady=(0, 40), padx=(0, 30))

# END RIGHT PANEL

root.mainloop()