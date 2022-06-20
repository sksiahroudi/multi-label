from tensorflow.keras import optimizers

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import  label_ranking_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from keras.layers import Dropout
import keras 
import scipy
from scipy.spatial import distance
from tensorflow.python.keras.metrics import Metric
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score, recall_score
from scipy.stats import ortho_group
import tensorflow as tf
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import hamming_loss

def Scene_model(train_x,train_y,epochs, batch_size,Optimizer):
    model = Sequential()
    LR=0.001
    Optimizer = tf.optimizers.Adam(lr=0.001)
    dim_x,dim_y=train_x.shape
    
    model.add(Dense(256, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size,verbose=0) 
    return model,LR

def F_exam(Predicted,Ground_truth,class_size): 
    precision=0
    recall=0
    for i in range (len(Predicted)):
        TN,FP,FN,TP= confusion_matrix(Ground_truth[i], Predicted[i]).ravel()

        if (TP+FP>0):
            precision=precision+(TP/(TP+FP))
        if(TP+FN>0):
            recall=recall+(TP/(TP+FN))
                
    precision=precision/len(Predicted)
    recall=recall/len(Predicted)
        
    if(precision+recall==0):
        return (0)
    F_example=2*(precision*recall)/(precision+recall)
    return F_example


def MicF_score(Predicted,Ground_truth,number_of_label):
    total_FP=0
    total_FN=0
    total_TP=0
    for i in range (number_of_label):
        
        TN,FP,FN,TP= confusion_matrix(Ground_truth[i], Predicted[i]).ravel()

        total_FP=total_FP+FP
        total_FN=total_FN+FN
        total_TP=total_TP+TP            
    Face=(2*total_TP)
    denominator=Face+(total_FP+total_FN)
    MicF=Face/denominator
        
    if (denominator==0):
        return 0
    print("total_TP",total_TP,"total_FP",total_FP,"total_FN",total_FN)
    return MicF
    
    
    
def MacF_score(Predicted,Ground_truth,number_of_label):
    
    Listof_FP=[]
    Listof_FN=[]
    Listof_TP=[]
    for i in range (number_of_label):
        TN,FP,FN,TP= confusion_matrix(Ground_truth[i], Predicted[i]).ravel()
        Listof_FP.append(FP)
        Listof_FN.append(FN)
        Listof_TP.append(TP)


    sumi=0
    Face=0
    Denominator=0
    for i in range (number_of_label):
        Face=2*Listof_TP[i]
        Denominator=Listof_FP[i]+Listof_FN[i]+2*Listof_TP[i]
        if (Denominator==0):
            continue
        sumi=sumi+(Face/Denominator)   

    MacF=sumi/number_of_label
    
    return MacF



def F_exam(Predicted,Ground_truth,class_size): 
    precision=0
    recall=0
    print(len(Predicted[0]))
    for i in range (len(Predicted[0])):
        TN,FP,FN,TP= confusion_matrix(Ground_truth[:,i], Predicted[:,i], labels=[0,1]).ravel()


        if (TP+FP>0):
            precision=precision+(TP/(TP+FP))
        if(TP+FN>0):
            recall=recall+(TP/(TP+FN))
                
    precision=precision/len(Predicted[0])
    recall=recall/len(Predicted[0])
        
    if(precision+recall==0):
        return (0)
    F_example=2*(precision*recall)/(precision+recall)
    return F_example


def MicF_score(Predicted,Ground_truth,number_of_label):
    total_FP=0
    total_FN=0
    total_TP=0
    for i in range (number_of_label):
        
        TN,FP,FN,TP= confusion_matrix(Ground_truth[i], Predicted[i], labels=[0,1]).ravel()
        if(TP+FP>0 and TP+FN>0 ):
            pprecision=TP/(TP+FP)
            rrecall=TP/(TP+FN)
            FF=0
            if(pprecision+rrecall>0):
                FF= 2*(pprecision*rrecall)/(pprecision+rrecall)        
            print(i,"==MicF===",FF,"==pprecision=",pprecision,"==rrecall==",rrecall)
        total_FP=total_FP+FP
        total_FN=total_FN+FN
        total_TP=total_TP+TP            
    Face=(2*total_TP)
    denominator=Face+(total_FP+total_FN)
    MicF=Face/denominator
        
    if (denominator==0):
        return 0
    return MicF
    
    
    
def MacF_score(Predicted,Ground_truth,number_of_label):
    
    Listof_FP=[]
    Listof_FN=[]
    Listof_TP=[]
    for i in range (number_of_label):
        TN,FP,FN,TP= confusion_matrix(Ground_truth[i], Predicted[i], labels=[0,1]).ravel()
        Listof_FP.append(FP)
        Listof_FN.append(FN)
        Listof_TP.append(TP)


    sumi=0
    Face=0
    Denominator=0
    for i in range (number_of_label):
        Face=2*Listof_TP[i]
        Denominator=Listof_FP[i]+Listof_FN[i]+2*Listof_TP[i]
        if (Denominator==0):
            continue
        sumi=sumi+(Face/Denominator)   

    MacF=sumi/number_of_label
    
    return MacF



def Triming(DATASET_NUMBER):
    
    file_name=["emotions.csv","scene.csv","enron.csv","yeast.csv","medical.csv","CAL500.csv","bibtex.csv","Corel5k.csv","Birds.csv","flags.csv","yelp.csv","mediamill.csv"]
    DataSet_LABEL=[]
    LABEL_Emotions=['amazed', 'happy','relaxing','quiet','sad','angry','Class']
    LABEL_Scene=['Beach','Sunset','FallFoliage','Field','Mountain','Urban','Class']
    LABEL_Enron=['A_A8','C_C9','B_B12','C_C11','C_C5','C_C7','B_B2','B_B3','D_D16','A_A7','D_D1','A_A4','C_C2','A_A3','A_A1','D_D9','D_D19','B_B8','D_D12','D_D6','C_C8','A_A6','B_B9','A_A5','C_C10','B_B1','D_D5','B_B11','D_D2','B_B4','D_D15','C_C4','D_D8','B_B6','D_D3','D_D13','D_D7','C_C12','B_B7','C_C6','B_B5','D_D11','A_A2','C_C3','D_D10','D_D18','B_B13','D_D17','B_B10','C_C1','D_D4','C_C13','D_D14','Class']
    LABEL_Yest=['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10','Class11','Class12','Class13','Class14','Class']
    LABEL_Medical=['Class-0-593_70','Class-1-079_99','Class-2-786_09','Class-3-759_89','Class-4-753_0','Class-5-786_2','Class-6-V72_5','Class-7-511_9','Class-8-596_8','Class-9-599_0','Class-10-518_0','Class-11-593_5','Class-12-V13_09','Class-13-791_0','Class-14-789_00','Class-15-593_1','Class-16-462','Class-17-592_0','Class-18-786_59','Class-19-785_6','Class-20-V67_09','Class-21-795_5','Class-22-789_09','Class-23-786_50','Class-24-596_54','Class-25-787_03','Class-26-V42_0','Class-27-786_05','Class-28-753_21','Class-29-783_0','Class-30-277_00','Class-31-780_6','Class-32-486','Class-33-788_41','Class-34-V13_02','Class-35-493_90','Class-36-788_30','Class-37-753_3','Class-38-593_89','Class-39-758_6','Class-40-741_90','Class-41-591','Class-42-599_7','Class-43-279_12','Class-44-786_07','Class']
    LABEL_CAL500=["Angry-Agressive","NOT-Emotion-Angry-Agressive","Emotion-Arousing-Awakening","NOT-Emotion-Arousing-Awakening","Emotion-Bizarre-Weird","NOT-Emotion-Bizarre-Weird","Emotion-Calming-Soothing","NOT-Emotion-Calming-Soothing","Emotion-Carefree-Lighthearted","NOT-Emotion-Carefree-Lighthearted","Emotion-Cheerful-Festive","NOT-Emotion-Cheerful-Festive","Emotion-Emotional-Passionate","NOT-Emotion-Emotional-Passionate","Emotion-Exciting-Thrilling","NOT-Emotion-Exciting-Thrilling","Emotion-Happy","NOT-Emotion-Happy","Emotion-Laid-back-Mellow","NOT-Emotion-Laid-back-Mellow","Emotion-Light-Playful","NOT-Emotion-Light-Playful","Emotion-Loving-Romantic","NOT-Emotion-Loving-Romantic","Emotion-Pleasant-Comfortable","NOT-Emotion-Pleasant-Comfortable","Emotion-Positive-Optimistic","NOT-Emotion-Positive-Optimistic","Emotion-Powerful-Strong","NOT-Emotion-Powerful-Strong","Emotion-Sad","NOT-Emotion-Sad","Emotion-Tender-Soft","NOT-Emotion-Tender-Soft","Emotion-Touching-Loving","NOT-Emotion-Touching-Loving","Genre--_Alternative","Genre--_Alternative_Folk","Genre--_Bebop","Genre--_Brit_Pop","Genre--_Classic_Rock","Genre--_Contemporary_Blues","Genre--_Contemporary_RandB","Genre--_Cool_Jazz","Genre--_Country_Blues","Genre--_Dance_Pop","Genre--_Electric_Blues","Genre--_Funk","Genre--_Gospel","Genre--_Metal-Hard_Rock","Genre--_Punk","Genre--_Roots_Rock","Genre--_Singer-Songwriter","Genre--_Soft_Rock","Genre--_Soul","Genre--_Swing","Genre-Bluegrass","Genre-Blues","Genre-Country","Genre-Electronica","Genre-Folk","Genre-Hip_Hop-Rap","Genre-Jazz","Genre-Pop","Genre-RandB","Genre-Rock","Genre-World","Instrument_-_Acoustic_Guitar","Instrument_-_Ambient_Sounds","Instrument_-_Backing_vocals","Instrument_-_Bass","Instrument_-_Drum_Machine","Instrument_-_Drum_Set","Instrument_-_Electric_Guitar_(clean)","Instrument_-_Electric_Guitar_(distorted)","Instrument_-_Female_Lead_Vocals","Instrument_-_Hand_Drums","Instrument_-_Harmonica","Instrument_-_Horn_Section","Instrument_-_Male_Lead_Vocals","Instrument_-_Organ","Instrument_-_Piano","Instrument_-_Samples","Instrument_-_Saxophone","Instrument_-_Sequencer","Instrument_-_String_Ensemble","Instrument_-_Synthesizer","Instrument_-_Tambourine","Instrument_-_Trombone","Instrument_-_Trumpet","Instrument_-_Violin-Fiddle","Song-Catchy-Memorable","NOT-Song-Catchy-Memorable","Song-Changing_Energy_Level","NOT-Song-Changing_Energy_Level","Song-Fast_Tempo","NOT-Song-Fast_Tempo","Song-Heavy_Beat","NOT-Song-Heavy_Beat","Song-High_Energy","NOT-Song-High_Energy","Song-Like","NOT-Song-Like","Song-Positive_Feelings","NOT-Song-Positive_Feelings","Song-Quality","NOT-Song-Quality","Song-Recommend","NOT-Song-Recommend","Song-Recorded","NOT-Song-Recorded","Song-Texture_Acoustic","Song-Texture_Electric","Song-Texture_Synthesized","Song-Tonality","NOT-Song-Tonality","Song-Very_Danceable","NOT-Song-Very_Danceable","Usage-At_a_party","Usage-At_work","Usage-Cleaning_the_house","Usage-Driving","Usage-Exercising","Usage-Getting_ready_to_go_out","Usage-Going_to_sleep","Usage-Hanging_with_friends","Usage-Intensely_Listening","Usage-Reading","Usage-Romancing","Usage-Sleeping","Usage-Studying","Usage-Waking_up","Usage-With_the_family","Vocals-Aggressive","Vocals-Altered_with_Effects","Vocals-Breathy","Vocals-Call_and_Response","Vocals-Duet","Vocals-Emotional","Vocals-Falsetto","Vocals-Gravelly","Vocals-High-pitched","Vocals-Low-pitched","Vocals-Monotone","Vocals-Rapping","Vocals-Screaming","Vocals-Spoken","Vocals-Strong","Vocals-Vocal_Harmonies","Genre-Best--_Alternative","Genre-Best--_Classic_Rock","Genre-Best--_Metal-Hard_Rock","Genre-Best--_Punk","Genre-Best--_Soft_Rock","Genre-Best--_Soul","Genre-Best-Blues","Genre-Best-Country","Genre-Best-Electronica","Genre-Best-Folk","Genre-Best-Hip_Hop-Rap","Genre-Best-Jazz","Genre-Best-Pop","Genre-Best-RandB","Genre-Best-Rock","Genre-Best-World","Instrument_-_Acoustic_Guitar-Solo","Instrument_-_Electric_Guitar_(clean)-Solo","Instrument_-_Electric_Guitar_(distorted)-Solo","Instrument_-_Female_Lead_Vocals-Solo","Instrument_-_Harmonica-Solo","Instrument_-_Male_Lead_Vocals-Solo","Instrument_-_Piano-Solo","Instrument_-_Saxophone-Solo","Instrument_-_Trumpet-Solo","Class"]
    LABEL_Bibtex=['TAG_2005','TAG_2006','TAG_2007','TAG_agdetection','TAG_algorithms','TAG_amperometry','TAG_analysis','TAG_and','TAG_annotation','TAG_antibody','TAG_apob','TAG_architecture','TAG_article','TAG_bettasplendens','TAG_bibteximport','TAG_book','TAG_children','TAG_classification','TAG_clustering','TAG_cognition','TAG_collaboration','TAG_collaborative','TAG_community','TAG_competition','TAG_complex','TAG_complexity','TAG_compounds','TAG_computer','TAG_computing','TAG_concept','TAG_context','TAG_cortex','TAG_critical','TAG_data','TAG_datamining','TAG_date','TAG_design','TAG_development','TAG_diffusion','TAG_diplomathesis','TAG_disability','TAG_dynamics','TAG_education','TAG_elearning','TAG_electrochemistry','TAG_elisa','TAG_empirical','TAG_energy','TAG_engineering','TAG_epitope','TAG_equation','TAG_evaluation','TAG_evolution','TAG_fca','TAG_folksonomy','TAG_formal','TAG_fornepomuk','TAG_games','TAG_granular','TAG_graph','TAG_hci','TAG_homogeneous','TAG_imaging','TAG_immunoassay','TAG_immunoelectrode','TAG_immunosensor','TAG_information','TAG_informationretrieval','TAG_kaldesignresearch','TAG_kinetic','TAG_knowledge','TAG_knowledgemanagement','TAG_langen','TAG_language','TAG_ldl','TAG_learning','TAG_liposome','TAG_litreview','TAG_logic','TAG_maintenance','TAG_management','TAG_mapping','TAG_marotzkiwinfried','TAG_mathematics','TAG_mathgamespatterns','TAG_methodology','TAG_metrics','TAG_mining','TAG_model','TAG_modeling','TAG_models','TAG_molecular','TAG_montecarlo','TAG_myown','TAG_narrative','TAG_nepomuk','TAG_network','TAG_networks','TAG_nlp','TAG_nonequilibrium','TAG_notag','TAG_objectoriented','TAG_of','TAG_ontologies','TAG_ontology','TAG_pattern','TAG_patterns','TAG_phase','TAG_physics','TAG_process','TAG_programming','TAG_prolearn','TAG_psycholinguistics','TAG_quantum','TAG_random','TAG_rdf','TAG_representation','TAG_requirements','TAG_research','TAG_review','TAG_science','TAG_search','TAG_semantic','TAG_semantics','TAG_semanticweb','TAG_sequence','TAG_simulation','TAG_simulations','TAG_sna','TAG_social','TAG_socialnets','TAG_software','TAG_spin','TAG_statistics','TAG_statphys23','TAG_structure','TAG_survey','TAG_system','TAG_systems','TAG_tagging','TAG_technology','TAG_theory','TAG_topic1','TAG_topic10','TAG_topic11','TAG_topic2','TAG_topic3','TAG_topic4','TAG_topic6','TAG_topic7','TAG_topic8','TAG_topic9','TAG_toread','TAG_transition','TAG_visual','TAG_visualization','TAG_web','TAG_web20','TAG_wiki','Class']
    LABEL_Corel5k=["city","mountain","sky","sun","water","clouds","tree","bay","lake","sea","beach","boats","people","branch","leaf","grass","plain","palm","horizon","shell","hills","waves","birds","land","dog","bridge","ships","buildings","fence","island","storm","peaks","jet","plane","runway","basket","flight","flag","helicopter","boeing","prop","f-16","tails","smoke","formation","bear","polar","snow","tundra","ice","head","black","reflection","ground","forest","fall","river","field","flowers","stream","meadow","rocks","hillside","shrubs","close-up","grizzly","cubs","drum","log","hut","sunset","display","plants","pool","coral","fan","anemone","fish","ocean","diver","sunrise","face","sand","rainbow","farms","reefs","vegetation","house","village","carvings","path","wood","dress","coast","sailboats","cat","tiger","bengal","fox","kit","run","shadows","winter","autumn","cliff","bush","rockface","pair","den","coyote","light","arctic","shore","town","road","chapel","moon","harbor","windmills","restaurant","wall","skyline","window","clothes","shops","street","cafe","tables","nets","crafts","roofs","ruins","stone","cars","castle","courtyard","statue","stairs","costume","sponges","sign","palace","paintings","sheep","valley","balcony","post","gate","plaza","festival","temple","sculpture","museum","hotel","art","fountain","market","door","mural","garden","star","butterfly","angelfish","lion","cave","crab","grouper","pagoda","buddha","decoration","monastery","landscape","detail","writing","sails","food","room","entrance","fruit","night","perch","cow","figures","facade","chairs","guard","pond","church","park","barn","arch","hats","cathedral","ceremony","crowd","glass","shrine","model","pillar","carpet","monument","floor","vines","cottage","poppies","lawn","tower","vegetables","bench","rose","tulip","canal","cheese","railing","dock","horses","petals","umbrella","column","waterfalls","elephant","monks","pattern","interior","vendor","silhouette","architecture","blossoms","athlete","parade","ladder","sidewalk","store","steps","relief","fog","frost","frozen","rapids","crystals","spider","needles","stick","mist","doorway","vineyard","pottery","pots","military","designs","mushrooms","terrace","tent","bulls","giant","tortoise","wings","albatross","booby","nest","hawk","iguana","lizard","marine","penguin","deer","white-tailed","horns","slope","mule","fawn","antlers","elk","caribou","herd","moose","clearing","mare","foals","orchid","lily","stems","row","chrysanthemums","blooms","cactus","saguaro","giraffe","zebra","tusks","hands","train","desert","dunes","canyon","lighthouse","mast","seals","texture","dust","pepper","swimmers","pyramid","mosque","sphinx","truck","fly","trunk","baby","eagle","lynx","rodent","squirrel","goat","marsh","wolf","pack","dall","porcupine","whales","rabbit","tracks","crops","animals","moss","trail","locomotive","railroad","vehicle","aerial","range","insect","man","woman","rice","prayer","glacier","harvest","girl","indian","pole","dance","african","shirt","buddhist","tomb","outside","shade","formula","turn","straightaway","prototype","steel","scotland","ceiling","furniture","lichen","pups","antelope","pebbles","remains","leopard","jeep","calf","reptile","snake","cougar","oahu","kauai","maui","school","canoe","race","hawaii","Class"]
    LABEL_Birds=["Brown Creeper","Pacific Wren","Pacific-slope Flycatcher","Red-breasted Nuthatch","Dark-eyed Junco","Olive-sided Flycatcher","Hermit Thrush","Chestnut-backed Chickadee","Varied Thrush","Hermit Warbler","Swainson Thrush","Hammond Flycatcher","Western Tanager","Black-headed Grosbeak","Golden Crowned Kinglet","Warbling Vireo","MacGillivray Warbler","Stellar Jay","Common Nighthawk","Class"]
    LABEL_Flag=["red","green","blue","yellow","white","black","orange","Class"]
    LABEL_Yelp=["IsFoodGood","IsServiceGood","IsAmbianceGood","IsDealsGood","IsPriceGood","Class"]
    LABEL_Mediamill=["Class1","Class2","Class3","Class4","Class5","Class6","Class7","Class8","Class9","Class10","Class11","Class12","Class13","Class14","Class15","Class16","Class17","Class18","Class19","Class20","Class21","Class22","Class23","Class24","Class25","Class26","Class27","Class28","Class29","Class30","Class31","Class32","Class33","Class34","Class35","Class36","Class37","Class38","Class39","Class40","Class41","Class42","Class43","Class44","Class45","Class46","Class47","Class48","Class49","Class50","Class51","Class52","Class53","Class54","Class55","Class56","Class57","Class58","Class59","Class60","Class61","Class62","Class63","Class64","Class65","Class66","Class67","Class68","Class69","Class70","Class71","Class72","Class73","Class74","Class75","Class76","Class77","Class78","Class79","Class80","Class81","Class82","Class83","Class84","Class85","Class86","Class87","Class88","Class89","Class90","Class91","Class92","Class93","Class94","Class95","Class96","Class97","Class98","Class99","Class100","Class101","Class"]
    DataSet_LABEL.append(LABEL_Emotions )
    DataSet_LABEL.append(LABEL_Scene )
    DataSet_LABEL.append(LABEL_Enron )
    DataSet_LABEL.append(LABEL_Yest)
    DataSet_LABEL.append(LABEL_Medical)
    DataSet_LABEL.append(LABEL_CAL500)
    DataSet_LABEL.append(LABEL_Bibtex)
    DataSet_LABEL.append(LABEL_Corel5k)
    DataSet_LABEL.append(LABEL_Birds)
    DataSet_LABEL.append(LABEL_Flag)
    DataSet_LABEL.append(LABEL_Yelp)
    DataSet_LABEL.append(LABEL_Mediamill)
        
    DataSet_dim_y=[]
    dim_y_Emotions=72
    dim_y_Scene=294
    dim_y_Yest=103
    dim_y_Enron=1001
    dim_y_Medical=1449
    dim_y_CAL500=68
    dim_y_Bibtex=1836
    dim_y_Corel5k=499
    dim_y_Birds=260
    dim_y_Flag=19
    dim_y_Yelp=671
    dim_y_Mediamill=120
    DataSet_dim_y.append(dim_y_Emotions)
    DataSet_dim_y.append(dim_y_Scene)
    DataSet_dim_y.append(dim_y_Enron)
    DataSet_dim_y.append(dim_y_Yest)
    DataSet_dim_y.append(dim_y_Medical)
    DataSet_dim_y.append(dim_y_CAL500)
    DataSet_dim_y.append(dim_y_Bibtex)
    DataSet_dim_y.append(dim_y_Corel5k)
    DataSet_dim_y.append(dim_y_Birds)
    DataSet_dim_y.append(dim_y_Flag)
    DataSet_dim_y.append(dim_y_Yelp)
    DataSet_dim_y.append(dim_y_Mediamill)
    
    DataSet_PATH=[]
    PATH_Emotions="../Emotions"
    PATH_Scene="../Scene"
    PATH_Enron="../Enron"
    PATH_Yeast="../Yeast"
    PATH_Medical="../Medical"
    PATH_CAL500="../CAL500"
    PATH_Bibt="../Bibtex"
    PATH_Corel5k="../Corel5k"
    PATH_Birds="../Birds"
    PATH_Flag="../../Flag"
    PATH_Yelp="../Yelp"
    PATH_Mediamill="../Mediamill"
    DataSet_PATH.append(PATH_Emotions)
    DataSet_PATH.append(PATH_Scene)
    DataSet_PATH.append(PATH_Enron)
    DataSet_PATH.append(PATH_Yeast)
    DataSet_PATH.append(PATH_Medical)
    DataSet_PATH.append(PATH_CAL500)
    DataSet_PATH.append(PATH_Bibt)
    DataSet_PATH.append(PATH_Corel5k)
    DataSet_PATH.append(PATH_Birds)
    DataSet_PATH.append(PATH_Flag)
    DataSet_PATH.append(PATH_Yelp)
    DataSet_PATH.append(PATH_Mediamill)
    
    DataSet_ClassSize=[]
    class_size_Emototions=6
    class_size_Scene=6
    class_size_Yeast=14
    class_size_Enron=53
    class_size_Medical=45
    class_size_CAL500=174
    class_size_Bibtex=159
    class_size_Corel5k=374
    class_size_Birds=19
    class_size_Flag=7
    class_size_Yelp=5
    class_size_Mediamill=101
    DataSet_ClassSize.append(class_size_Emototions)
    DataSet_ClassSize.append(class_size_Scene)
    DataSet_ClassSize.append(class_size_Enron)
    DataSet_ClassSize.append(class_size_Yeast)
    DataSet_ClassSize.append(class_size_Medical)
    DataSet_ClassSize.append(class_size_CAL500)
    DataSet_ClassSize.append(class_size_Bibtex)
    DataSet_ClassSize.append(class_size_Corel5k)
    DataSet_ClassSize.append(class_size_Birds)
    DataSet_ClassSize.append(class_size_Flag)
    DataSet_ClassSize.append(class_size_Yelp)
    DataSet_ClassSize.append(class_size_Mediamill)
    
    PATH=DataSet_PATH[DATASET_NUMBER]
    dim_y=DataSet_dim_y[DATASET_NUMBER]
    LABEL=DataSet_LABEL[DATASET_NUMBER]
    class_size=DataSet_ClassSize[DATASET_NUMBER]
    return class_size,PATH,dim_y,LABEL,file_name[DATASET_NUMBER]




def generate_synthetic(train_x_0,train_y_0,label_set,label):
    label_set2=label_set.copy()
    label_set2.remove("Class")
    train_x_0=train_x_0.drop("Class",axis=1)
    for i in  label_set2:
        train_x_0[i]=train_x_0[i]*0
        
        #print(train_x_0[i])
    #print("!!!!!!!!!!!!!!!!!!!!before!!!!!!!!!!!!!!",train_x_0.shape,"  train_y_0.sum():",train_y_0.sum())
    train_size_x,train_size_y=train_x_0.shape
    train_x=train_x_0
    train_y=train_y_0


    train_x=train_x.drop(label_set2,axis=1)
    return train_x,train_y


def learning_model_for_feature_selection(class_size,dim_y,train_x,train_y):
    
    X, y = make_friedman1(n_samples=class_size, n_features=dim_y, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector = selector.fit(train_x, train_y)
    selector.support_    
    return selector
    
    
def remove_feature(DATASET_NUMBER,xx,DATA,LABEL,dim_y):
    print("------------removing started----********************************************")
    remove_label_each_class=[]
    label_list=set([])
    class_size=len(LABEL)-1
    for k in range (class_size):
        #print("________________________________________________k=",k,label_list)
        train_x_0,train_y_0,ALL_data=read_data(LABEL[k],LABEL,DATA)
        #train_x,train_y = generate_synthetic(train_x_0,train_y_0)
        LABEL_SET_Without_current=LABEL.copy()
        LABEL_SET_Without_current.remove(LABEL[k])
        #LABEL_SET_Without_current.remove("Class")
        #train_x,train_y = generate_synthetic2(ALL_data,train_y_0,LABEL_SET_Without_current,LABEL[k],orthogonal_matrix,i,train_size_x)
        train_x,train_y = generate_synthetic(ALL_data,train_y_0,LABEL_SET_Without_current,LABEL[k])

        selector=learning_model_for_feature_selection(class_size,dim_y,train_x,train_y)
        j=0
        removed=[]
        if (k==0):
            label_list=set(selector.ranking_)
        for i in range (1,dim_y):
            if (selector.ranking_[i]>dim_y/xx):
               # print(i)
                j=j+1
                removed.append(i)
        label_list=set.intersection(set(removed), set(label_list))
        remove_label_each_class.append(removed)
        print(k,"____________len(removing_features)___________",len(label_list))
    aa=list(label_list)
    removing_features=[]
    for i in range (len(aa)):
        removing_features.append(train_x.columns[aa[i]])
    
    return removing_features


def dist_select(xx,selected):
    tmp_dist=[]
    for i in range (len(xx)):
        tmp_dist.append(0)
        for j in selected:
            if (i==j):
                break
            tmp_dist[i]=tmp_dist[i]+scipy.spatial.distance.euclidean(xx[i],xx[j])
    ind=np.argmax(tmp_dist)
    selected.append(ind)
    return selected
    
def generate_ortho(dim,times,train_x_0):
    x=[]
    m = train_x_0.mean(axis = 0, skipna = False)
    mm=np.array(m)
    tmp=[]
    xx=[]
    sumi=[]
    TMP=[]
    for i in range (times*30):
        tmp.append(ortho_group.rvs(dim))
        xx.append(inner_produc(mm, tmp[i],1))
        
        xx[i]=xx[i].flatten()
    
    selected=[]
    selected.append(0)
    for k in range (times):
        selected=dist_select(xx,selected)
          
    for i in range(times):
        x.append(tmp[selected[i]])        
    return x
def Enron_model(train_x,train_y,epochs, batch_size,Optimizer):
    model = Sequential()
    LR=0.001
    Optimizer = tf.optimizers.Adam(lr=0.001)
    dim_x,dim_y=train_x.shape
    model.add(Dense(32, input_dim=dim_y, activation='relu'))#294


    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size, verbose=1) 
    return model,LR

def Birds_model(train_x,train_y,epochs, batch_size,Optimizer):
    LR=0.001
    model = Sequential()
   
    Optimizer = tf.optimizers.Adam(learning_rate=LR)
    dim_x,dim_y=train_x.shape
    

    model.add(Dense(512, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size, verbose=1) 
    return model,LR
def Yeast_model(train_x,train_y,epochs, batch_size,Optimizer):
    model = Sequential()
    LR=0.001
    Optimizer = tf.optimizers.Adam(lr=0.001)
    dim_x,dim_y=train_x.shape
    
    model.add(Dense(256, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    mcp_save = keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')

    model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')

    model.fit(X, y,epochs, batch_size, callbacks=[es_callback, mcp_save])
    #model.fit(X, y,epochs, batch_size) 
    return model,LR
def Mediamill_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) :
    model = Sequential()
    LR=0.001
    Optimizer = keras.optimizers.Adam(learning_rate=LR)
    dim_x,dim_y=train_x.shape
    model.add(Dense(1024, input_dim=dim_y, activation='relu'))#294

    #model.add(Dense(512, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size, verbose=1) 
    return model,LR
def Flag_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) :
    model = Sequential()
    LR=0.001
    Optimizer = tf.optimizers.Adam(lr=0.001)
    dim_x,dim_y=train_x.shape
    
    model.add(Dense(64, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size) 
    return model,LR

def model (TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) :
    if (DATASET_NUMBER==0):
        return Emotion_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) 
    if (DATASET_NUMBER==1):
        return Scene_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) 
    if(DATASET_NUMBER==2):
        return Enron_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) 
    if(DATASET_NUMBER==3):
        return Yeast_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) 
    if(DATASET_NUMBER==8):
        return Birds_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer)
    if(DATASET_NUMBER==9):
        return Flag_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer)
    if(DATASET_NUMBER==11):
        return Mediamill_model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer)
def Emotion_model(train_x,train_y,epochs, batch_size,Optimizer):
    model = Sequential()
    LR=0.001
    Optimizer = optimizers.Adam(lr=0.001)
#    Optimizer = tf.optimizers.Adam(lr=0.001)
    dim_x,dim_y=train_x.shape
    
    model.add(Dense(64, input_dim=dim_y, activation='relu'))#294
    
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=["accuracy"])
    #model.compile(optimizer=Optimizer,              loss='mse',               metrics=["accuracy"])
    X=train_x
    y=train_y
    #es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    #model.fit(X, y,epochs, batch_size,callbacks=[es_callback])
    model.fit(X, y,epochs, batch_size,verbose=0) 
    return model,LR

def read_data(label,set_LABEL,DATA):
    data=DATA.copy()
    data['Class']=0
    data['Class'][data[label] == 1] = 1
    ALL_data=data.drop(label,axis=1)

    y=data['Class']
    #print("++++++++++++++++++++++++++++++++",label)
    data=data.drop(set_LABEL,axis=1)
   
    return data,y,ALL_data

def read_test_data(set_LABEL,DATA):
    data=DATA
    set_LABEL2=set_LABEL.copy()
    set_LABEL2.remove("Class")
    Y=data[set_LABEL2]
    X=data.drop(set_LABEL2,axis=1)
    return X,Y

def inner_produc(data, C, index):
   
    data2=np.array(data)
    data2=np.inner(data,C) ## data,c or data2,c?
    return data2



 


def run_train(removing_features,TRAIN_DATA,index,LABEL):   
    
    class_size=len(LABEL)-1
    ortho=[]
    removed_labels=[]
    flag=0
    
    train_x_0,train_y_0,ALL_data=read_data(LABEL[0],LABEL,TRAIN_DATA)
    print("00000000000000000000000000000:",len(removing_features))
    if (len (removing_features)>1):
        train_x_0=train_x_0.drop(removing_features,axis=1)
    train_size_x,dim_y=train_x_0.shape
    
    orthogonal_matrix=generate_ortho(dim_y,class_size,train_x_0)   

    for i in range (0,class_size):
        sum_removed_class=0
        #
        train_x_0,train_y_0,ALL_data=read_data(LABEL[i],LABEL,TRAIN_DATA)
        if (len (removing_features)>1):
            train_x_0=train_x_0.drop(removing_features,axis=1)
            ALL_data=ALL_data.drop(removing_features,axis=1)
        #print("------------------train_x_0.shape----------",train_x_0.shape,"ALL_data_shape",ALL_data.shape)
        train_size_x,dim_y=train_x_0.shape
        LABEL_SET_Without_current=LABEL.copy()
        LABEL_SET_Without_current.remove(LABEL[i])
        #LABEL_SET_Without_current.remove("Class")
        train_x,train_y =generate_synthetic(ALL_data,train_y_0,LABEL_SET_Without_current,LABEL[i])
 
        


        train_x_m=inner_produc(train_x,orthogonal_matrix[i], train_size_x)
    
        if (flag==0):
            flag=1+flag
            TRAIN_x=train_x_m
            TRAIN_y=train_y
        else:
            
            TRAIN_x=np.concatenate((TRAIN_x,train_x_m), axis=0)
            TRAIN_y=np.concatenate((TRAIN_y,train_y), axis=0)

    return sum_removed_class,TRAIN_x,TRAIN_y,removed_labels,orthogonal_matrix




def run_test(m,removed_labels,ortho,removing_features,TEST_DATA,index,LABEL):
    
   



    x_REAL,Y_REAL0=read_test_data(LABEL,TEST_DATA)
    if (len (removing_features)>1):
        x_REAL=x_REAL.drop(removing_features,axis=1)
    
    Y_PREDICT=Y_REAL0.copy()
    j=0
    Valid_Label=[]
    class_size=len(LABEL)-1
    PREDICTION=[]
 
    Grounrd_truth=[]
    for i in range (class_size):

        test_x,test_y,_=read_data(LABEL[i],LABEL,TEST_DATA)
        if (len (removing_features)>1):
            test_x=test_x.drop(removing_features,axis=1)
        
        test_size_x,test_size_y=test_x.shape
        test_x_m=inner_produc(np.array(test_x),ortho[j],test_size_x)
        pre=m.predict(test_x_m)

        Y_PREDICT[LABEL[i]]=pre
        Y=np.array(test_y)
        Y_REAL=np.array(test_y)
 
        PREDICTION.append(pre)

        Grounrd_truth.append(Y)
    
        

        j=j+1




    
    return(class_size,Grounrd_truth,PREDICTION)

def drop_rare_label(DATA,LABEL):
    lb=[]
    a,b=DATA.shape
    for label in LABEL:

        if label=="Class":
            lb.append(label)
            continue
        #print(label,")))))))))))))))))))))))))))))))))))))))",DATA[label].sum())
        if (DATA[label].sum()<20 or a/DATA[label].sum() >=50):
            #print(label,"0000000000000000000000000000000000000000000000",DATA[label].sum())
            DATA=DATA.drop(label , axis=1)
            
        else:
            print(label,"________________________",DATA[label].sum())
            lb.append(label)
    print("DATA.shape-------------------after:-----------" ,DATA.shape)
    return DATA,lb
       
from sklearn.decomposition import PCA

def cv_read(path,LABEL,file_name): 
    print(path+"/"+file_name)
    df=pd.read_csv(path+"/"+file_name ,",")
       
    
    
    train=pd.read_csv(path+"/train.csv")
    test=pd.read_csv(path+"/test.csv")
       
    df,lb=drop_rare_label(df,LABEL)
    print("*******************************************label size",len(lb), "feature size:",df.shape)
   
    
    train.reset_index(inplace=True)
    train=train.drop("index",axis=1)
    
    test.reset_index(inplace=True)
    test=test.drop("index",axis=1)
    print("label size",len(lb))
    print("train shape",train.shape)
    print("test_shape",test.shape)

    return train,test,lb


def save_RM_features(path,removing_features,i):
    f2 = open(path+"/rm_features"+str(i)+".csv", "w")
    for ii in range (len(removing_features)):
        f2.write(str(removing_features[ii]))
        if(ii<len(removing_features)-1):
            f2.write(",")
    f2.close()


    
def read_rm_features(path,i):
    f = open(path+"/rm_features"+str(i)+".csv", "r")
    xx=f.read()
    x = xx.split(",")
  
    return x


def print_res_to_file(i,jj,PATH,Example_F1,global_F_micro,AVG_F_macroo,sum_removed_class,removing_features,batch_size,epochs,rm,index):
    f = open("RESULT_with_feature_selection.txt", "a")
    f.write("\n \n Dropout(0.5)      path:"+str(PATH))
    f.write("\n trial number: iteration number:="+str(i)+":"+"   ensemble number:"+str(jj))
    f.write("\n Example_F1:"+str(Example_F1))
    f.write("\n microF1:"+str(np.mean(global_F_micro)))
    f.write("\n MacroF1:"+str(np.mean(AVG_F_macroo)))
    f.write("\n sum_removed_class"+str(sum_removed_class))
    f.write("\n removing_features size"+str(len(removing_features)))
    f.write("\n batch_size::"+str(batch_size)+"     epochs:"+str(epochs)+"  feature percentage (rm):"+str(rm)+"  index:"+str(index))

    f.write("\n ###################################################################### \n \n ")
    f.close()
    
    
    
from scipy.spatial import distance
def distance(data,i,j):
    return scipy.spatial.distance.euclidean(data[i],data[j])

def common_ele(a, b):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) == 0:
        return True
    return False

def neighbors_detection(TRAIN_x,TRAIN_y,kk_index):
    #return TRAIN_x,TRAIN_y
    print("Shape :",(TRAIN_x.shape))
    print("number of positive: ",TRAIN_y.sum())
    print("type TRAIN_x :",type(TRAIN_x))
    print("type of TRAIN_y: ",type(TRAIN_y))
    kk=kk_index
    row,col=TRAIN_x.shape
    removed_list=[]
    
    dist_array = np.empty((row, row), float)
    #label_array =np.empty((1, row), float)
    for i in range (row):
        for j in range (row):
            dist_array[i][j]=100000
            
            
    for i in range (row):

        for j in range (i+1,row):
            dist_array[i,j]=distance(TRAIN_x,i,j)
            dist_array[j,i]=dist_array[i,j]  
        if(TRAIN_y[i]==1):
            continue 
        idx = np.argpartition(dist_array[i], kk)
        min_indeces=   idx[0:kk]
        
        if (common_ele(removed_list,min_indeces)):
            sum_label=0
            for ii in (min_indeces):
                sum_label +=TRAIN_y[ii]
            if(sum_label==0):
                removed_list.append(i)
                dist_array[:,i]=100000

      
        #print(dist_array[i],"\n")
    print("neighbors_detection done with kk_index:",kk_index,TRAIN_x.shape)
    TRAIN_x=np.delete(TRAIN_x, removed_list, 0)
    TRAIN_y=np.delete(TRAIN_y, removed_list, 0)
    print("neighbors_detection done with kk_index:",kk_index,TRAIN_x.shape)
    return TRAIN_x,TRAIN_y
        

    

def write_result(path1,path2,PRE_total,Grounrd_truth,class_size,res,TRAIN_x,TRAIN_y,a1,a2, PRE_total_real):
    
        MICF=MicF_score(PRE_total,Grounrd_truth,class_size) 
        MACF=MacF_score( PRE_total,Grounrd_truth,class_size)
        Exam_F=F_exam(PRE_total,Grounrd_truth,class_size)
        hammingloss=hamming_loss(Grounrd_truth, PRE_total)
            
        coverageerror1=coverage_error(Grounrd_truth, PRE_total)
        coverageerror2=coverage_error(Grounrd_truth, res)
        aucMac_traons=roc_auc_score(np.transpose(Grounrd_truth), np.transpose(res), average='macro')
        try:
            aucMac_normal=roc_auc_score(Grounrd_truth,res, average='macro')
        except ValueError:
            aucMac_normal=0
        try:
            aucMac_normal_total=roc_auc_score(Grounrd_truth,PRE_total, average='macro')
        except ValueError:
            aucMac_normal_total=0       
        try:    
            AVG_pers_mac_norm=average_precision_score( Grounrd_truth,res,  average='macro')
        except ValueError:
            AVG_pers_mac_norm=0
                
        AVG_pers_mac_trans=average_precision_score( np.transpose(Grounrd_truth), np.transpose(res),  average='macro')
        
        AVG_pers_mac_trans_total=average_precision_score( np.transpose(Grounrd_truth), np.transpose(PRE_total_real),  average='macro')
        
        ranking2=label_ranking_loss(Grounrd_truth, res)
        ranking_trans=label_ranking_loss(np.transpose(Grounrd_truth), np.transpose(res))
        f = open(path1, "a")
            
        f.write("\n \n ------------------------------------------------------------")
        f.write("\n MICF:"+str(MICF)+" MACF:"+str(MACF)+"  Exam_F "+str(Exam_F)+"  hamming_loss"+str(hammingloss)+" ranking2:"+str(ranking2)+
                  " AVG_pers_mac_norm:"+str(AVG_pers_mac_norm)+" AVG_pers_mac_trans: "+str(AVG_pers_mac_trans)+
                  " aucMac_traons: "+str(aucMac_traons)+"  aucMac_normal: "+str(aucMac_normal)+"\n")
            
        f.write("TRAIN_x.shape"+str(TRAIN_x.shape)+"  TRAIN_y.sum: "+str(TRAIN_y.sum()) +" removed:"+str(a1-a2))
        f.close()
           
        f2 = open(path2, "a")
        f2.write("\n"+str(kk_index)+",   "+str(MICF)+","+str(MACF)+","+str(Exam_F)+","+str(hammingloss)+","+
                 str(ranking2)+","+str(ranking_trans)+","+str(AVG_pers_mac_norm)+","+str(AVG_pers_mac_trans)+","+str(aucMac_normal)
                +","+str(aucMac_traons)+","+str(a1-a2))
        f2.close()
            
def main(path1,path2,DATASET_NUMBER,batch_size,epochs,removing_flag,rm,taghsim,LR,iterations,ensemble_size,kk_index):


        PRE_list=[]
           
        removing_features=[]
        
 
            
        optimizer="Adam"#"SGD","Adadelta", "Adam" "Adamax"
        class_size,PATH,dim_y,LABEL,file_name=Triming(DATASET_NUMBER)
        TRAIN_data,TEST_data,LABEL  =cv_read(PATH,LABEL,file_name)
       
        
        
           
            
        if i==0 and removing_flag==True:
            removing_features=remove_feature(DATASET_NUMBER,rm,TRAIN_data,LABEL,dim_y)
            save_RM_features(PATH,removing_features,rm)
        
        removing_features=read_rm_features(PATH,rm)
          
         
          
        for jj in range (iterations):
            sum_removed_class,TRAIN_x,TRAIN_y,removed_labels,ortho=run_train(removing_features,TRAIN_data,DATASET_NUMBER,LABEL)
            a1,b1=TRAIN_x.shape
            TRAIN_x,TRAIN_y=neighbors_detection(TRAIN_x,TRAIN_y,kk_index)
            print("\\n\\n\\n\\n\\n",TRAIN_x.shape,"\\n\\n\\n\\n",TRAIN_y.shape,"\\n\\n\\n\\n")
            m,LR=model(TRAIN_x,TRAIN_y, batch_size,epochs,optimizer) 
            class_size,Grounrd_truth,predictions=run_test(m,removed_labels,ortho,removing_features,TEST_data,DATASET_NUMBER,LABEL)
            PRE_list.append(predictions)   
            
            if jj==0:
                PRE_total=np.array(predictions)
            else:
                PRE_total=np.array(predictions)+PRE_total
        
        res=PRE_total.copy()
        for t in range(1):
            taghsim=taghsim
            r,c=TEST_data.shape
            res=np.reshape(res,(class_size,r))
            

         
            
            print("ground::\n")
           


            
            
            PRE_total_real=res/taghsim
            PRE_total=np.round(PRE_total_real)
        
            a2,b2=TRAIN_x.shape
            
            write_result(path1,path2,PRE_total,Grounrd_truth,class_size,res,TRAIN_x,TRAIN_y,a1,a2, PRE_total_real)
                
     

           
            
path1="result/Flag_journal_ortho.txt"
path2="result/Flag_journal_ortho.csv"


thisdict = {"ensemble_size":5,
  "taghsim": 3.8,
  "DATASET_NUMBER": 11,
    "RM": 5.0,
  "LR":0.001,
  "batchSize": 16,"epochsSize": 40,
  "removing_flag": False,
  "iterations": 5, "kk_index":50
    
}

f2 = open(path2, "a")
f2.write("\n index,MICF,MACF,Exam_F,hammingloss,rankingLoss,rankingLoss_trans,AVG_pers_mac_norm,AVG_pers_mac_trans,aucMac_normal,aucMac_traons,removed")
f2.close()
for i in range (1,100):
    thisdict["kk_index"]=int(i*3)
    ensemble_size=thisdict["ensemble_size"]
    taghsim=thisdict["taghsim"]
    print(taghsim)
    DATASET_NUMBER=thisdict["DATASET_NUMBER"]
    RM=thisdict["RM"]
    LR=thisdict["LR"]
    batchSize=thisdict["batchSize"]nano
    epochsSize=thisdict["epochsSize"]
    removing_flag=thisdict["removing_flag"]
    iterations=thisdict["iterations"]
    kk_index=thisdict["kk_index"]
    main (path1,path2,DATASET_NUMBER,batchSize,epochsSize,removing_flag,RM,taghsim,LR,iterations,ensemble_size,kk_index)




    f = open(path1, "a")

    for k, v in thisdict.items():
        f.write("  "+str(k) + ':'+ str(v)+"  " )
        
    f.close()   
