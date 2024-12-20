#-------------------------------------------------------------------------------------------#
#                               IMPORTING THE REQUIRED LIBRARIES                            #
#-------------------------------------------------------------------------------------------#

from sklearn import datasets, neighbors, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------------------#
#   WRITING A FUNCTION TO BUILD A MACHINE LEARNING MODEL BASED ON USER CHOICES IN THE GUI   #
#                                                                                           #
#                                        PARAMETERS-                                        #
#                                1. DATASET CHOSEN BY USER                                  #
#                           2. CLASSIFICATION MODEL CHOSEN BY USER                          #
#                                                                                           #                         
#                                           OUTPUTS-                                        #
#                                     1. CONFUSION MATRIX                                   #
#                                   2. CLASSIFICATION REPORT                                #
#                      3. PLOTS FOR PARAMETER WHILE RUNNING CROSS-VALIDATION                #
#                                        4. ACCURACY                                        #
#                                    5. BEST PARAMETERS                                     #
#                                     6. GRID SCORES                                        #
#-------------------------------------------------------------------------------------------#

def build_model(dataset, cf_model):
    
    #Loading user's choice of data into a variable- dataset
    if dataset == "Iris":
        data = datasets.load_iris() 
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset == "Wine":
        data = datasets.load_wine() 
    
    #Exctracting values from the dataset and storing it into a variable X
    X = data.data 
    
    #Storing target values in y
    y = data.target
    
    #Extracting the class names present in the target variable
    class_names = data.target_names
    
    #-------------------------------------------------------------------------------------------#
    #               SPLITTING DATA INTO TRAINING AND TEST SETS IN 80:20 RATIO                   #
    #       (X_TRAIN, y_TRAIN) HAVING 80% OF DATA AND (X_TEST, y_TEST) HAVING 20% OF DATA       #
    #-------------------------------------------------------------------------------------------#
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    
    #-------------------------------------------------------------------------------------------#
    #                   BUILDING MODEL BASED ON USER'S CHOICE OF CLASSIFIER                     #
    #-------------------------------------------------------------------------------------------#
    if cf_model == "KNN":
        #Setting a range of parameters
        param_range = [{'n_neighbors': [1, 2, 3, 4, 5]}]
        model = neighbors.KNeighborsClassifier()

    elif cf_model == "SVM":
        #Setting a range of parameters
        param_range = [{'C':[0.0001, 0.001, 0.01, 0.1, 1.0]}]
        model = svm.SVC()

    elif cf_model == "GMM":
        #Setting a range of parameters
        param_range = [{'n_components': [1, 2, 3, 4, 5]}]  
        model = GaussianMixture(covariance_type='diag')

    #-------------------------------------------------------------------------------------------#   
    #                   PERFORMING GRID SEARCH CROSS-VALIDATION WITH K = 5                      #
    #-------------------------------------------------------------------------------------------#
    gscv_model = GridSearchCV(
    estimator = model,
    param_grid = param_range,
    cv = 5,
    scoring = "accuracy")

    #-------------------------------------------------------------------------------------------#
    #                           TRAINING THE MODEL ON BEST PARAMETERS                           #
    #-------------------------------------------------------------------------------------------#
    gscv_model.fit(X_train, y_train)
    
    #-------------------------------------------------------------------------------------------#
    #           Getting all the required parameters from the GridSearch model                   #
    #-------------------------------------------------------------------------------------------#
    mean_test_score = gscv_model.cv_results_['mean_test_score']
    stds_test_score = gscv_model.cv_results_['std_test_score']
    results = gscv_model.cv_results_['params']
    
    #-------------------------------------------------------------------------------------------#
    #                       GETTING THE BEST PARAMETERS OF THE MODEL                            #                      
    #-------------------------------------------------------------------------------------------#
    best_parameter = gscv_model.best_params_
    
    #-------------------------------------------------------------------------------------------#
    #                                  TESTING THE MODEL                                        #
    #-------------------------------------------------------------------------------------------#
    y_pred = gscv_model.predict(X_test)
    
    #Getting the accuracy
    accuracy = metrics.accuracy_score(y_test,y_pred) * 100

    #-------------------------------------------------------------------------------------------#
    #                               PLOTTING CONFUSION MATRIX                                   #
    #-------------------------------------------------------------------------------------------#
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    plotcm = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = class_names)
    plt.title('Confusion Matrix Plot')
    plotcm.plot()
    plt.savefig("image1.png", dpi = 70)
    plt.clf()
    
    #-------------------------------------------------------------------------------------------#
    #                           Creating a cross-validation plot for KNN                        #
    #-------------------------------------------------------------------------------------------#
    if cf_model == "KNN":
        k_range = [1, 2, 3, 4, 5]
        plt.plot(k_range, gscv_model.cv_results_['mean_test_score'])
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        plt.title('Cross Validation plot for KNN')
        plt.savefig('image2.png', dpi=70)
        plt.clf()

    #-------------------------------------------------------------------------------------------#
    #               Creating a cross-validation plot for GMM using BIC scores                   #
    #-------------------------------------------------------------------------------------------#  
    elif cf_model == "GMM":
        bic_scores = []
        n_components_range = [1, 2, 3, 4, 5]
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
            gmm.fit(X_train)  
            bic_scores.append(gmm.bic(X_train))  

        #Creating a bar plot to visualize BIC scores
        plt.bar(n_components_range, bic_scores)
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Score')
        plt.title('BIC Score for GMM Models')
        plt.savefig('image2.png', dpi=70)
        plt.clf()

    #-------------------------------------------------------------------------------------------#
    #                       Creating a cross-validation plot for SVM                            #
    #-------------------------------------------------------------------------------------------#
    elif cf_model == "SVM":
        C_s = [0.0001, 0.001, 0.01, 0.1, 1.0]
        plt.semilogx(C_s, np.array(mean_test_score))
        plt.semilogx(C_s, np.array(mean_test_score) + np.array(stds_test_score), 'b--')
        plt.semilogx(C_s, np.array(mean_test_score) - np.array(stds_test_score), 'b--')
        locs, labels = plt.yticks()
        plt.yticks(locs,list(map(lambda x: "%g" % x, locs)))
        plt.ylabel('CV score')
        plt.xlabel('Parameter C')
        plt.title('Cross Validation Plot for SVM')
        plt.savefig("image2.png",dpi = 70)
        plt.clf()
    
    plt.close("all")
    
    #-------------------------------------------------------------------------------------------#
    #     Creating a dictionary to store all the values calculated above to be shown in GUI     #
    #-------------------------------------------------------------------------------------------#
    final_output = dict(                       
        class_names = class_names,  
        y_test = y_test,                                           
        y_pred = y_pred,                
        best_params = best_parameter, 
        mean_test_score = mean_test_score,
        stds_test_score = stds_test_score,  
        accuracy = accuracy, 
        results = results,
        confusion_matrix = confusion_matrix)   
    
    return final_output

