from PIL import Image
import pytesseract
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# features = []
# started = False
# for filename in tqdm(os.listdir("/Users/thomaswoodside/Dropbox/Dank Memes/Put "
#                             "memes here")):
#     text = pytesseract.image_to_string(Image.open(
#         "/Users/thomaswoodside/Dropbox/Dank Memes/Put memes here/" +
#         filename)).replace("\n", " ")
#     if not started and text == "intrude z unescapabm n  "YOU'LL SEE!!!! THEY'LL ALL SEE!!!"  7 a passiona'e eye doctor as he throws glasses mm a screammg crowd (we richarcl)  _ ack'ﬁ:     Some 53; ,  390,763 noﬁes ... >  U.":
#         started = True
#         continue
#     if started and text == 'jackson @tncyc'eichamp  [god making pandas]  ' \
#                            'GOD: cow bears  ANGEL: what  GOD: did ifucken stuﬂer  ANGEL:  GOD: take it a cow and make it a bear 3:38 AM - 15 May 2015  4' 13 390 V 618  y Follow':
#         started = False
#         continue
#     if not started:
#         features.append(text)
#     print(features, started)
#
#
# print(features)
# vect = TfidfVectorizer(stop_words="english")
# features = vect.fit_transform(features)
# pickle.dump(vect, open("vect.pkl", "wb"))
# pickle.dump(features, open("features.pkl", "wb"))
# features = pickle.load(open('features.pkl', 'rb'))
# labels = [["Political", "That escalated quickly"], ["Dirty", "Weird"], [], ["Dirty"], ["Dirty", "Don't know if..."], ["Nerdy"], ["Politics"], ["If I had a dollar"], ['That escalated quickly'], ["Sports"], ['Parents'], [], ['Parents'], ["Today", "Pokemon"], ['Happy'], ['Irony'], [''], ['Today'], ['One does not simply'], ['Nerdy', "One does not simply"], [], ["USA"], [], ["Don't know if...", "Race"], ["Parents", "That escalated quickly"], ['Conspiracy', "Don't know if...", 'Stupid'], ["Stupid", 'Gender difference'], ['Gender difference'], ['Nerdy'], [], ['School'], ['Today'], [], [], ["Weird", "Gender difference"], ['Gender difference', 'Politics'], ['Irony'], ['Dirty'], [], ['Irony', 'Stupid'], ['One does not simply'], [], ['Gender difference'], ["Gender difference", 'Irony'], ["Irony"], ['Conspiracy'], ['Irony'], [], ['Dirty'], [], ["Stupid"], ['Irony'], ["Politics"], ['Irony'], ["Irony"], [], ['Irony'], ["Pokemon"], ["Irony"], ["Irony", 'Depression'], ['Nerdy'], ['Dirty'], ['Gender difference'], ['Dirty','Gender Difference'], ['Irony', 'Stupid'], [], ['Dirty'], ['Irony'], [], [], ['Sesame street','Terrorism'], ['One does not simply'], ['Terrorism'], ["Nerdy", 'Gender difference'], ['Irony'], ['Pokemon'], ['Funny', 'Weird'], ["Political"], ["Gender difference"], ['Irony'], ['Sesame street'], ['Depression', 'Terrorism'], ['Irony'], ['Irony', 'Today'], [], ['4chan'], ['One does not simply'], ['Irony'], ['Dirty'], ['Today','Parents'], [], ['Gender difference"'], ['Squad'], [], ["Stupid"], ["4chan"],[], ['Pokemon','Irony'], ['Irony'], ["Dirty"], ['Sesame street', 'Terrorism'], [], ['Pokemon'], ['Irony'], ['Tinder'], ["Sesame street", 'Terrorism'], ['Today', 'Parents'], ['USA',"Countries"], ['Today'], ['Tinder'], ['Nerdy', 'Stupid'], ["Stupid", "Irony"], ["Stupid"], [], ["Nerdy", "Irony"], ["Funny", "Nerdy", "Sports"], ["Funny", "Irony"], [], ['Terrorism','Gender difference'], ["Stupid"], [], [], ["Terrorism"], ['Irony'], [], [], [], ['Terrorism','Sesame street'], ['Pokemon'], ["Funny", "Irony"], ["Irony", "Funny", "Stupid"], [], ['Irony'], [], [], ['Irony','Gender difference'], ['Irony'], [], ["Pokemon"],['Dat Boi'], ['Tinder'], [], [], [], ['Sesame street','Terrorism'], ['Irony'], [], [], ['Irony'], [], [], ['Gender difference'], ['Sesame street','Terrorism'], ['Irony'], [], [], ['Irony','Dirty'], ['USA'], [], ['Sesame street','Terrorism'], ['Squad'], [], ['Tinder'], [], [],[],['Parents'], ['Dat Boi'], ["Dirty"], [], ["Countries"], ['Irony'], ['Irony'], [], ["Sesame street","Terrorism"], ['Political'], [], [], ["Irony"], ['Stupid'], ["Parents"], [], [], ['Parents'], [], ["Irony"], ['Parents'], ['Today'], [], ['Dirty', "Sesame street"], ['USA', 'Countries'], ['Stupid'], [], ["Sesame street", "Terrorism"], ['Irony'], [], ['Irony'], ['Today'], ['Irony'], ["Today"], ['Sesame street'], [], [], ['Today'], ["Depression"], [], ['Today'], ['Dirty'], ['Gender difference'], ["Today"], ['Today', 'Dogo'], ["Today", 'Gender difference'], ['Gender difference'], ['Gender difference', 'Depressing', 'John Cena', 'One does not simply'], [], ["Irony"], ['Sesame street'], ["That escalated quickly", 'One does not simply'], ['Dirty'], ['Parents'], ['Dirty'], ['Sesame street'], ['Parents'], ['Irony'], ["Irony"], [], ["Drugs", "Irony"], [], ['Irony'], ['Dirty'], [], ['Irony'], [], [], ["Gender differences"], ["Gender differences"], ['Politics', 'Dirty'], ["Tinder"], [], ["Tinder"], [], [], ["Gender difference"], ["Dirty"], ['Irony'], ['Dirty'], ['Dirty'], ['Terrorism'], [], ["Tinder"], ["Tinder"], [], [], [], [], [], ['Irony','Gender difference'], [], ["Dirty"], [], ["Dirty"], [], [], ['Drugs'], [], ['Depression'], ["Gender difference"], [], ['Parents'], ["Gender differences"], [], ["Sesame street"], ['Face swap'], [], [], ['Pokemon'], ['Sesame street'], [], ['Terrorism','Conspiracy'], ['Stupid','Funny'], ["Cop"], ["Sesame street"], [], ['Funny'], ["Funny"],["Gender difference"], ["Pokemon"], ["Squad"], ["Funny"], ["Funny"], ["Countries"], [],  ["Tinder"], ["Parents"], ["Funny"], [], ["dirty"], ["Funny"], ["Funny"], ["political"], ["Terrorism","Funny"]]
# mlb = MultiLabelBinarizer()
# labels = mlb.fit_transform(labels)
# pickle.dump(mlb, open("mlb.pkl", "wb"))
# pca = PCA(n_components='mle')
# features = pca.fit_transform(features.toarray())
# pickle.dump(mlb, open("pca.pkl", "wb"))
# clf = RandomForestClassifier(class_weight="balanced", n_estimators=100)
# print("clf")
# clf.fit(features, labels)
# print(clf.best_params_)
# pickle.dump(clf, open("clf.pkl", "wb"))


vect = pickle.load(open("vect.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))

text = [pytesseract.image_to_string(Image.open(
         "Sample_Memes/SRi1yp3.jpg")).replace("\n", " ")]

print(text)
predictions = clf.predict_proba(vect.transform(text))
max = 0
maxname = ''
for prediction, name in zip(predictions, mlb.classes_):
    print(prediction)
    if prediction[0][1] > max:
        maxname = name
        max = prediction[0][1]
print(maxname)
