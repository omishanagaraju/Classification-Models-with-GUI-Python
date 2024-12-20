#-------------------------------------------------------------------------------------------#
#                               IMPORTING THE REQUIRED LIBRARIES                            #
#-------------------------------------------------------------------------------------------#

import os
import tkinter as tk
from tkinter import PhotoImage, messagebox
from tkinter.constants import BOTH, YES
import algorithms_module as algorithm

#-------------------------------------------------------------------------------------------#
#                          DEFINING VISUALS OF THE MAIN WINDOW                              #
#-------------------------------------------------------------------------------------------#

screen = tk.Tk() 

#Setting the title of the window
screen.title("Classification GUI") 

#Setting window size and dimensions with 3 rows and 2 columns
screen.geometry("1330x820+300+100")
screen.rowconfigure(3, minsize=10, weight=1)  
screen.columnconfigure(2, minsize=50, weight=1) 

#Selecting font, size, color and adding a header
screen_font = "Calibri, 15"
screen_header = tk.Label(text="MACHINE LEARNING MODELS FOR CLASSIFICATION", font="Calibri, 20", height=3, fg="darkmagenta")
screen_header.grid(row = 0, column = 0, columnspan = 2)

#Creating aframe
screen_frame = tk.Frame(master=screen, width=500, height=430, bg="floral white")
screen_frame.grid(row=1, column=0, sticky="W")

#Creating widget to display visual components
screen_widget = tk.Canvas(screen, bg="white", width=1000, height=430)
screen_widget.grid(row=1, column=1, sticky="NSEW")

#Adding a label for displaying the Confusion matrix
matrix_label = tk.Label(screen_widget, bg="white")
matrix_label.place(x=40,y=40)

#Adding a label for displaying the Cross Validation Plot 
CV_label = tk.Label(screen_widget, bg="white")
CV_label.place(x=540,y=40)

#Adding a label to display the accuracy of the ML model
show_accuracy_label = tk.Label(screen, fg="lightseagreen", font="Calibri, 18", anchor="nw")
show_accuracy_label.place(x=480, y=510)

#Adding a label to show the best parameter calculated using Grid Search CV
show_parameter_label = tk.Label(screen, fg="purple", font=screen_font, anchor="nw")
show_parameter_label.place(x=20, y=565)

#Adding a label to display the confusion matrix
confusion_matrix_label = tk.Label(screen, fg="orchid", anchor="nw", font=screen_font)
confusion_matrix_label.place(x=20, y=620)

#Adding a label to display the cross validation results
show_cross_val_label = tk.Label(screen, fg="seagreen", font=screen_font, anchor="nw")
show_cross_val_label.place(x=20, y=760)

#Adding a label to show class names on screen
show_class_label = tk.Label(screen, fg="royalblue", font=screen_font, anchor="nw")
show_class_label.place(x=850, y=575)

#Adding a label to display the test score
test_values_label = tk.Label(screen, fg="maroon", anchor="nw", font=screen_font)
test_values_label.place(x=850, y=620)

#Adding a label to display predicted values for the chosen ML model
prediction_value_label = tk.Label(screen, fg="indianred", anchor="nw", font=screen_font, justify="left")
prediction_value_label.place(x=850, y=750)


#-------------------------------------------------------------------------------------------#
#                   CREATING UI WITH RADIO BUTTONS TO CHOOSE THE DATASET                    #
#-------------------------------------------------------------------------------------------#

#Adding label for choosing the dataset
dataset = tk.Label(master=screen_frame, text="PLEASE CHOOSE THE DATASET",
                          fg="teal", anchor="w", width=70, height=1, font=screen_font,
                          bg="floral white")
dataset.place(x=10, y=10) 

show_dataset_label = tk.Label(master=screen_frame, text="", fg="firebrick", anchor="w", width=50, 
                         height=1, font=screen_font, bg="floral white")


data = tk.StringVar()  
data.set(' ') 

#Function to display chosen dataset
def choosen_data():
    output = "The " + str(data.get()) + " Dataset is chosen." 
    show_dataset_label.config(text=output)

#Adding button to allow user to choose the Iris Dataset
iris_button = tk.Radiobutton(master=screen_frame, text="Iris Dataset", value="Iris",
                           variable=data, command=choosen_data,
                           bg="floral white", font=screen_font)
iris_button.place(x=15, y=40)

#Adding button to allow user to choose the Breast Cancer Dataset 
breastcancer_button = tk.Radiobutton(master=screen_frame, text="Breast Cancer Dataset",
                                   variable=data, command=choosen_data,
                                   value="Breast Cancer", bg="floral white", font=screen_font)
breastcancer_button.place(x=15, y=70)

#Adding button to allow user to choose the Wine Dataset
wine_button = tk.Radiobutton(master=screen_frame, text="Wine Data", value="Wine",
                           variable=data, command=choosen_data,
                           bg="floral white", font=screen_font)
wine_button.place(x=15, y=100)

show_dataset_label.place(x=15, y=130) 

#-------------------------------------------------------------------------------------------#
#                CREATING UI WITH RADIO BUTTONS TO CHOOSE THE CLASSIFIER                    #
#-------------------------------------------------------------------------------------------#

#Adding label to get user's choice of Classifier
classifier_label = tk.Label(master=screen_frame, text="PLEASE SELECT THE MODEL FOR CLASSIFICATION",
                                fg="teal", anchor="w", width=70, height=1, font=screen_font,
                                bg="floral white")
classifier_label.place(x=10, y=180)

show_classifier_label = tk.Label(master=screen_frame, text="", fg="firebrick", anchor="w", width=50, 
                         height=1, font=screen_font, bg="floral white")

#Defining string variable
classifier = tk.StringVar()
classifier.set(' ')

#Function to display the chosen classifier
def chosen_classifier():
    classifier_op = "The " + str(classifier.get()) + " Classifier is chosen."
    show_classifier_label.config(text=classifier_op)
    
#Adding radio button to allow user to select the K-Nearest Neighbours Classifier
knn_button = tk.Radiobutton(master=screen_frame, text="KNN",
                          variable=classifier,
                          command=chosen_classifier,
                          value="KNN", bg="floral white", font=screen_font)
knn_button.place(x=15, y=210)

##Adding radio button to allow user to select the Guassian Mixture Model Classifier
button_GMM = tk.Radiobutton(master=screen_frame, text="GMM",
                          variable=classifier,
                          command=chosen_classifier,
                          value="GMM", bg="floral white", font=screen_font)
button_GMM.place(x=15, y=240)

##Adding radio button to allow user to select the Support Vector Machine Classifier 
svm_button = tk.Radiobutton(master=screen_frame, text="SVM",
                          variable=classifier,
                          command=chosen_classifier,
                          value="SVM", bg="floral white", font=screen_font)
svm_button.place(x=15, y=270) 

show_classifier_label.place(x=15, y=300)

#-------------------------------------------------------------------------------------------#
#                RUNNING ML ALGORITHM BASED ON USER'S CHOICE OF DATA AND MODEL              #
#-------------------------------------------------------------------------------------------#

#DEFINING FUNCTION
def ml_algorithm():

    #Storing the dataset, model and number of K-Folds chosen by user in seperate variables
    selected_dataset = data.get()
    selected_classifier = classifier.get()
    
    #Printing rules needed to execute the algorithm
    alert = "UNABLE TO RUN THE PROGRAM\n"
    alert += "\n PLEASE SELECT-\n"
    
    #Displaying error message if any of the elements are not selected by the user
    counter = 0
    if selected_dataset == " ":
        counter += 1
        alert += "\n" + str(counter) +  ". DATASET\n"
    
    if selected_classifier == " ":
        counter += 1
        alert += "\n" + str(counter) +  ". CLASSIFICATION MODEL\n"
        
    if selected_dataset != " " and selected_classifier != " ":
        
        #Building classification model to display the required output based on user's choice of data, mofel and k-folds
        ml_model = algorithm.build_model(selected_dataset,selected_classifier)
        
        #Getting path of the file
        dir_name = os.path.dirname(__file__)
        
        #Getting Confusion Matrix plot of model, saved in image1 using it's path
        image_1 = "image1.png" 
        file_1 = os.path.join(dir_name, image_1)
        
        #Creating an object of the confusion matrix image to display it on the GUI 
        cf_matrix = PhotoImage(file = file_1)
        matrix_label.image = cf_matrix
        matrix_label.configure(image = cf_matrix)
        
        #Getting the cross validation plot saved in a file named image2 
        image_2 = "image2.png" 
        file_2 = os.path.join(dir_name, image_2)
        img = PhotoImage(file = file_2)
        CV_label.image = img
        CV_label.configure(image = img)
        
        #-------------------------------------------------------------------------------------------#
        #                            DISPLAYING REQUIRED OUTPUT ON GUI                              #
        #-------------------------------------------------------------------------------------------#

        #Showing class labels present in the data
        show_class_label.config(text="CLASS NAMES- " + str(ml_model['class_names']))
        
        #Outputting the best paramters
        show_parameter_label.config(text="\nBEST PARAMETER-  " + str(ml_model['best_params']))
        
        #Outputting the accuracy of the ml model
        show_accuracy_label.config(text="\nMODEL ACCURACY- {0:.2f}%".format(ml_model['accuracy']))

        #Outputting the cross-validation scores of the ml model
        means = ml_model['mean_test_score']
        results = ml_model['results']
        stds = ml_model['stds_test_score']
        
        grid_scores = "GRID SCORES ON VALIDATION SET-\n"
        for mean, param , std in zip(means, results, stds):
            grid_scores += "Parameter: %r, accuracy: %0.3f (+/-%0.03f)\n" % (param, mean, std*2)

        show_cross_val_label.config(text=grid_scores)

        #Outputting the prediction value of the model
        prediction_value_label.config(text="PREDICTED VALUES- " + str(ml_model['y_pred']))
        
        #Outputting the test value of model
        test_values_label.config(text="\nACTUAL VALUES- " + str(ml_model['y_test'])) 

        #Outputting the confusion matrix for the model
        confusion_matrix_label.config(text = "\nCONFUSION MATRIX-\n " + str(ml_model['confusion_matrix']))

    else:
        #Displaying alert    
        messagebox.showwarning("Warning", alert)
        screen.focus_force()


#-------------------------------------------------------------------------------------------#
#                                   CREATING A RUN BUTTON                                 #
#-------------------------------------------------------------------------------------------#

run_button = tk.Button(master=screen_frame, text="RUN", fg="forestgreen",
                   bg="lavender", width=10, height=1, font=screen_font,
                   command=ml_algorithm)
run_button.place(x=30, y=360)

#-------------------------------------------------------------------------------------------#
#                                   CREATING A RESET BUTTON                                 #
#-------------------------------------------------------------------------------------------#

#Defining a function to clear various elements in the GUI and make it ready for new input 
def reset_ip():
    data.set(' ')
    show_dataset_label.config(text="")
    
    classifier.set(' ')
    show_classifier_label.config(text="")
    
    matrix_label.config(image="")
    CV_label.config(image="")
    
    show_class_label.config(text="")
    test_values_label.config(text="")

    show_parameter_label.config(text="")
    show_accuracy_label.config(text="")
    show_cross_val_label.config(text="")
    prediction_value_label.config(text="")
    confusion_matrix_label.config(text="")

#Creating the reset button
reset_button = tk.Button(master=screen_frame, text="RESET", fg="orange",
                     bg="lavender", width=10, height=1, font=screen_font,
                     command=reset_ip)
reset_button.place(x=190, y=360)

#-------------------------------------------------------------------------------------------#
#                                   CREATING AN QUIT BUTTON                                 #
#-------------------------------------------------------------------------------------------#

#Defining a function to quit application
def exit_app():
    screen.destroy()  

#Creating the quit button
quit_button = tk.Button(master=screen_frame, text= "QUIT", height=1,
                    width=15, font=screen_font, bg="lavender", fg="crimson",
                    command=exit_app)
quit_button.place(x=80, y=400)

#-------------------------------------------------------------------------------------------#
#                               DISPLAYING TKINTER SCREEN CREATED                           #
#-------------------------------------------------------------------------------------------#
screen.mainloop()
