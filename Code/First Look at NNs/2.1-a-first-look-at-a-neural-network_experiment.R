#This code is adapted from Deep Learning with R by Chollet & Allaire
#Dependencies:
#Anaconda

#Load package manager
if (!require("pacman")) install.packages("pacman")

pacman::p_load(keras,install = TRUE, update = FALSE)

#Install Keras. Only need to run this once on your computer. 
install_keras()

#Load data
mnist <- dataset_mnist() #This will download large dataset, cca 200Mb
n <- 60000
train_images <- array_reshape(mnist$train$x, c(n, 28 * 28)) / 255
n_test <- 10000
test_images <- array_reshape(mnist$test$x, c(n_test, 28 * 28)) / 255
train_labels <- to_categorical(mnist$train$y)
test_labels <- to_categorical(mnist$test$y)

#Make some plots
i <- sample(1:n, 1)
d <- array_reshape(train_images[i,],c(28,28))
plot(as.raster(d,max=1))
train_labels[i,]

#architecture
net <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")

#This function compiles, fits, and tests the net using default parameters
test <- function(net){
#compile
net %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#train
net %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)

#test
metrics <- net %>% evaluate(test_images, test_labels, verbose = 0)
metrics
}

#Test basic net
test(net)

#Generate some predictions
net %>% predict_classes(test_images[1:10,])

#Experiments
#Change number of epochs
net %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)

#Change batch size
net %>% fit(train_images, train_labels, epochs = 5, batch_size = 16)

#Change width
net <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")
test(net)

#Change number of layers
net <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")
test(net)

#Change width and number of layers
net <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")
test(net)
