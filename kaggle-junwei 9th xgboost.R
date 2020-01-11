#library
library(caret)
library(randomForest)
library(dplyr)
install.packages("xgboost")
library(xgboost)

# Read in analysis data and scoring data ---------------------------------------------------------------------------------------------
analysisData <- read.csv('analysisData.csv')
scoringData <- read.csv('scoringData.csv')

#set a new scoringdata in order to combine analysisData and scoringData----------------------------------------------------------------
scoringDataNew <- scoringData
scoringDataNew$price <- NA #since scoringData has no price, I create a new variable in scoringDataNew----------------------------------

# Use rbind to combine data together --------------------------------------------------------------------------------------------------
allDataSelected <- rbind(analysisData, scoringDataNew)

# Remove columns with inconsistent levels ---------------------------------
allDataSelected <- select(allDataSelected, -listing_url,-scrape_id,-last_scraped,-name,-summary,-space,-description,-experiences_offered,
                          -neighborhood_overview,-notes,-transit,-access,-interaction,-house_rules,-thumbnail_url,-medium_url,
                          -xl_picture_url, -host_id,-host_url,-host_name,-host_since,-host_about,-jurisdiction_names,-picture_url,
                          -host_acceptance_rate,host_thumbnail_url,-host_picture_url,-amenities,-first_review,-last_review,
                          -requires_license,-license,-country, -country_code,-has_availability,-state, -city, -host_location,
                          -market, -host_neighbourhood, -street, -smart_location, -host_thumbnail_url, -id, -host_response_time, 
                          -host_total_listings_count, -host_verifications, -neighbourhood, -neighbourhood_cleansed, -neighbourhood_group_cleansed
                          , -zipcode, -property_type, -bed_type, -weekly_price, -monthly_price, -square_feet, -calendar_updated, 
                          -calendar_last_scraped, -cancellation_policy, -host_response_rate)

#Set the levels for factors
levels(allDataSelected$host_is_superhost) <- c(0, 1)
levels(allDataSelected$host_has_profile_pic) <- c(0, 1)
levels(allDataSelected$host_identity_verified) <- c(0, 1)
levels(allDataSelected$is_location_exact) <- c(0, 1)
levels(allDataSelected$instant_bookable) <- c(0, 1)
levels(allDataSelected$is_business_travel_ready) <- c(0, 1)
levels(allDataSelected$require_guest_profile_picture) <- c(0, 1)
levels(allDataSelected$require_guest_phone_verification) <- c(0, 1)
levels(allDataSelected$room_type) <- c(3, 2, 1)

# Fill in missing values using preProcess method---------------------------------------------------------------------------------------
newDataClean <- predict(preProcess(allDataSelected, method = 'medianImpute'),
                        newdata = allDataSelected)

# Split into train and test data (train represents analysisData, and test represents scoringData)---------------------------------------
train <- newDataClean[1:29142,]
test <- newDataClean[29143:36428,]
colnames(train)

#set up xgboost model-------------------------------------------------------------------------------------------------------------
analysisPrice <- train$price
train <- select(train, -price)
test <- select(test, -price)

#convert to XG and put test and train data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(sapply(train, as.numeric)), label= analysisPrice)
dtest <- xgb.DMatrix(data = as.matrix(sapply(test, as.numeric)))

#set up default
default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.01,
  gamma=0,
  max_depth=8,
  min_child_weight=4, 
  subsample=1,
  colsample_bytree=1
)

#cross validation to get the best nrounds
xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 40, 
                 early_stopping_rounds = 10, maximize = F)

#fing the best nrounds and run the model
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 1200)

#Pred and see RMSE----------------------------------------------------------------------------------------------------------------------
XGBpred <- predict(xgb_mod, dtest)

#Submit
submissionFile = data.frame(id = scoringData$id, price = predictions_XGB)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)
getwd()
