# Adaboost

 An orientation line is created using the weighted means of the
 positive and negative points. Each point is then projected onto 
the line. the midpoint of each pair of projected points is then
calculated and used to create a separation boundary.
this boundary moved along the line calculating the number of point
 misclassifications each iteration. The best boundary then becomes the 
weak classifier.The weak classifier calculates its error in classification of points
 and uses this error to update the weights depending on whether the classification
was correct. Each weak classifier uses its error to calculate an indivdual alpha 
value associated with it. Adaboost_strong combines each weak classifier into a
strong classifier. Each weak classifier is weighted by their alpha values.
The accuracy of each weak classifier and the overall strong classifier is computed 
and the number of classifiers used is output. The strong classifer accuracy is output each
weak classifer iteration and the final strong classifier output is plotted  
