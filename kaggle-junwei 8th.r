#library
library(caret)
library(randomForest)

# Read in analysis data and scoring data ---------------------------------------------------------------------------------------------
analysisData <- read.csv('analysisData.csv')
scoringData <- read.csv('scoringData.csv')

#set a new scoringdata in order to combine analysisData and scoringData----------------------------------------------------------------
scoringDataNew <- scoringData
scoringDataNew$price <- NA #since scoringData has no price, I create a new variable in scoringDataNew----------------------------------

# Use rbind to combine data together --------------------------------------------------------------------------------------------------
newData <- rbind(analysisData, scoringDataNew)

# Fill in missing values using preProcess method---------------------------------------------------------------------------------------
newDataClean <- predict(preProcess(newData, method = 'medianImpute'),
                        newdata = newData)

# Split into train and test data (train represents analysisData, and test represents scoringData)---------------------------------------
train <- newDataClean[1:29142,]
test <- newDataClean[29143:36428,]

#set up random forest model-------------------------------------------------------------------------------------------------------------
finalForest = randomForest(price~host_listings_count + host_has_profile_pic + 
                        host_identity_verified  +  longitude + latitude +
                        is_location_exact + property_type + room_type + accommodates + 
                        bathrooms + bedrooms + beds +  cleaning_fee + square_feet +
                        guests_included + extra_people + minimum_nights + availability_30 + 
                        availability_365 + number_of_reviews + review_scores_rating + 
                        review_scores_cleanliness +  review_scores_communication + 
                        review_scores_location + review_scores_value  + 
                        cancellation_policy + require_guest_phone_verification + 
                        calculated_host_listings_count + reviews_per_month, data = train, ntree = 2000)

#Pred and see RMSE----------------------------------------------------------------------------------------------------------------------
predFinalForest = predict(finalForest, newdata = test)
rmseFinalForest = sqrt(mean((predFinalForest - train$price) ^ 2)); rmseFinalForest

submissionFile = data.frame(id = scoringData$id, price = predFinalForest)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)
getwd()
