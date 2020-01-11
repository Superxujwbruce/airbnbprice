library(tidyr); library(dplyr); library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(gbm)

#read two data files
analysisData <- read.csv('analysisData.csv')
scoringData <- read.csv('scoringData.csv')

#explore data
#name
dataNames <- colnames(analysisData);dataNames
#type
dataType <- sapply(analysisData, class);dataType
#which variables are factors
isFactor <- sapply(analysisData, is.factor);isFactor
#Check how many missing value
NaVariables <- sapply(analysisData, function(x) any(is.na(x)));NaVariables
CountNAs <- sapply(analysisData, function(x) sum(is.na(x)));CountNAs
#Check levels
numOfLevels <- sapply(analysisData, function(x) length(levels(x)))

# Remove useless variables --------------------------------------------
analysisData <- select(analysisData, -listing_url,-scrape_id,-last_scraped,-name,-summary,-space,-description,-experiences_offered,
                       -neighborhood_overview,-notes,-transit,-access,-interaction,-house_rules,-thumbnail_url,-medium_url,
                       -xl_picture_url, -host_id,-host_url,-host_name,-host_since,-host_about,-jurisdiction_names,-picture_url,
                       -host_acceptance_rate,host_thumbnail_url,-host_picture_url,-amenities,-first_review,-last_review,
                       -requires_license,-license,-country, -country_code,-has_availability,-state, -city, -host_location,
                       -market, -host_neighbourhood, -street, -smart_location)


# Fill in missing values (NAs) --------------------------------------------
analysisDataClean <- predict(preProcess(analysisData, method = 'medianImpute'),
                             newdata = analysisData)
scoringDataClean <- predict(preProcess(scoringData, method = 'medianImpute'),
                            newdata = scoringData)


#Find useful variables-Data Visualization---------------------------------------------------------------------------------------
#Correlation Plot---------------------------------------------------------------------------------------------------------------
#corrplot  <- analysisDataClean[,c('price','host_listings_count','latitude','longitude','accommodates','bathrooms')]
#pairs(corrplot,pch = 20,cex = 0.5,bg="green",col="blue",lwd= 0.4)

corrplot2  <- analysisDataClean[,c('price','review_scores_rating','review_scores_accuracy',
                                   'review_scores_cleanliness','review_scores_checkin','review_scores_location')]
pairs(corrplot,pch = 20,cex = 0.5,bg="green",col="blue",lwd= 0.4)

# Examine bivariate correlations ------------------------------------------
corData = analysisDataClean[sapply(analysisDataClean, class) == 'numeric' |  
                                  sapply(analysisDataClean, class) == 'integer' | 
                                  sapply(analysisDataClean, class) == 'logic']
corMatrix = as.data.frame(cor(corData))
corMatrix$var1 = rownames(corMatrix)
corMatrix %>%
  gather(key=var2,value=r,1:31)%>%
  ggplot(aes(x=var1,y=var2,fill=r))+
  geom_tile()+
  geom_text(aes(label=round(r,2)),size=3)+
  scale_fill_gradient2(low = 'red',high='green',mid = 'white')+
  theme(axis.text.x=element_text(angle=90))



#backward method
#Step:  AIC=241128.4 with zipcode
#price ~ host_is_superhost + host_identity_verified + zipcode + 
#  latitude + longitude + property_type + room_type + accommodates + 
# bathrooms + bedrooms + beds + square_feet + security_deposit + 
#  cleaning_fee + guests_included + minimum_nights + availability_30 + 
#  availability_90 + availability_365 + number_of_reviews + 
#  review_scores_rating + review_scores_accuracy + review_scores_cleanliness + 
#  review_scores_checkin + review_scores_location + review_scores_value + 
#  is_business_travel_ready + cancellation_policy + require_guest_phone_verification + 
#  calculated_host_listings_count + reviews_per_month + host_response_time + 
#  neighbourhood_group_cleansed + review_scores_communication

#Step:  AIC=243099.9 no zipcode
#price ~ host_is_superhost + host_identity_verified + latitude + 
#  longitude + property_type + room_type + accommodates + bathrooms + 
#  guests_included + minimum_nights + availability_30 + availability_90 + 
#  availability_365 + number_of_reviews + review_scores_rating + 
#  review_scores_cleanliness + review_scores_checkin + review_scores_location + 
#  review_scores_value + is_business_travel_ready + cancellation_policy + 
#  calculated_host_listings_count + reviews_per_month + host_response_time + 
# neighbourhood_group_cleansed + review_scores_communication

#Start:  AIC=243094.3 no zipcode
#price ~ host_is_superhost + host_has_profile_pic + host_identity_verified + 
#  latitude + longitude + property_type + room_type + accommodates + 
#  bathrooms + bedrooms + beds + square_feet + security_deposit + 
#  cleaning_fee + guests_included + minimum_nights + availability_30 + 
#  availability_90 + availability_365 + number_of_reviews + 
#  review_scores_rating + review_scores_cleanliness + review_scores_checkin + 
#  review_scores_location + review_scores_value + is_business_travel_ready + 
#  cancellation_policy + require_guest_phone_verification + 
#  calculated_host_listings_count + reviews_per_month + host_response_time + 
#  host_listings_count + neighbourhood_group_cleansed + extra_people + 
#  review_scores_communication

#Find useful variables--------------------------------------------
start_mod = lm(price~host_is_superhost + host_identity_verified + 
                   latitude + longitude + property_type + room_type + accommodates + 
                  bathrooms + bedrooms + beds + square_feet + security_deposit + 
                   cleaning_fee + guests_included + minimum_nights + availability_30 + 
                   availability_90 + availability_365 + number_of_reviews + 
                   review_scores_rating + review_scores_accuracy + review_scores_cleanliness + 
                   review_scores_checkin + review_scores_location + review_scores_value + 
                   is_business_travel_ready + cancellation_policy + require_guest_phone_verification + 
                   calculated_host_listings_count + reviews_per_month + host_response_time + 
                   neighbourhood_group_cleansed + review_scores_communication,data=analysisDataClean)
empty_mod = lm(price~1,analysisDataClean)
full_mod = lm(price~host_is_superhost + host_identity_verified + 
                latitude + longitude + property_type + room_type + accommodates + 
                bathrooms + bedrooms + beds + square_feet + security_deposit + 
                cleaning_fee + guests_included + minimum_nights + availability_30 + 
                availability_90 + availability_365 + number_of_reviews + 
                review_scores_rating + review_scores_accuracy + review_scores_cleanliness + 
                review_scores_checkin + review_scores_location + review_scores_value + 
                is_business_travel_ready + cancellation_policy + require_guest_phone_verification + 
                calculated_host_listings_count + reviews_per_month + host_response_time + 
                neighbourhood_group_cleansed + review_scores_communication,data=analysisDataClean)
backwardStepwise = step(start_mod,scope=list(upper=full_mod,lower=empty_mod),
                       direction='backward')
#another
start_mod = lm(price~host_is_superhost + host_has_profile_pic + host_identity_verified + 
                   latitude + longitude + property_type + room_type + accommodates + 
                  bathrooms + bedrooms + beds + square_feet + security_deposit + 
                   cleaning_fee + guests_included + minimum_nights + availability_30 + 
                   availability_90 + availability_365 + number_of_reviews + 
                   review_scores_rating + review_scores_cleanliness + review_scores_checkin + 
                   review_scores_location + review_scores_value + is_business_travel_ready + 
                   cancellation_policy + require_guest_phone_verification + 
                   calculated_host_listings_count + reviews_per_month + host_response_time + 
                   host_listings_count + neighbourhood_group_cleansed + extra_people + 
                   review_scores_communication,data=analysisDataClean)

empty_mod = lm(price~1,analysisDataClean)

full_mod = lm(price~host_is_superhost + host_has_profile_pic + host_identity_verified + 
                latitude + longitude + property_type + room_type + accommodates + 
                bathrooms + bedrooms + beds + square_feet + security_deposit + 
                cleaning_fee + guests_included + minimum_nights + availability_30 + 
                availability_90 + availability_365 + number_of_reviews + 
                review_scores_rating + review_scores_cleanliness + review_scores_checkin + 
                review_scores_location + review_scores_value + is_business_travel_ready + 
                cancellation_policy + require_guest_phone_verification + 
                calculated_host_listings_count + reviews_per_month + host_response_time + 
                host_listings_count + neighbourhood_group_cleansed + extra_people + 
                review_scores_communication,data=analysisDataClean)

backwardStepwise = step(start_mod,scope=list(upper=full_mod,lower=empty_mod),
                       direction='backward')




#as.factor Loop
for (p in colnames(analysisDataClean)) { 
  if (class(scoringDataClean[[p]]) == "factor") { 
    levels(scoringDataClean[[p]]) <- levels(analysisDataClean[[p]]) 
  }
}
#model--------------------------------------------------------

#Log Price----------------------------------------------------
analysisDataClean <- analysisDataClean[analysisDataClean$price>0,]
analysisDataClean$log_price <- log(analysisDataClean$price)
#Linear Regression--------------------------------------------
model <- lm(log_price~host_response_time+ host_listings_count +host_total_listings_count+
              neighbourhood_group_cleansed+availability_90+
              
              host_is_superhost + host_has_profile_pic + 
              host_identity_verified + longitude + latitude +
              is_location_exact + property_type + room_type + accommodates + 
              bathrooms + bedrooms + beds +  cleaning_fee + square_feet +
              guests_included + extra_people + minimum_nights + availability_30 + 
              availability_365 + number_of_reviews + review_scores_rating + 
              review_scores_cleanliness + review_scores_checkin + review_scores_communication + 
              review_scores_location + review_scores_value + is_business_travel_ready + 
              cancellation_policy + require_guest_phone_verification + 
              calculated_host_listings_count + reviews_per_month,data=analysisDataClean)

pred1 = predict(model,newdata=analysisDataClean)
pred2 <- exp(pred1);pred2
rmse1 = sqrt(mean((pred2-analysisDataClean$price)^2)); rmse1

#Random Forest--------------------------------------------
trControl=trainControl(method="cv",number=10)
tuneGrid = expand.grid(mtry=1:5)
set.seed(100)
cvForest = train(price~host_is_superhost + host_has_profile_pic + host_identity_verified + 
                   latitude + longitude + property_type + room_type + accommodates + 
                   bathrooms + bedrooms + beds + square_feet + security_deposit + 
                   cleaning_fee + guests_included + minimum_nights + availability_30 + 
                   availability_90 + availability_365 + number_of_reviews + 
                   review_scores_rating + review_scores_cleanliness + review_scores_checkin + 
                   review_scores_location + review_scores_value + is_business_travel_ready + 
                   cancellation_policy + require_guest_phone_verification + 
                   calculated_host_listings_count + reviews_per_month + host_response_time + 
                   host_listings_count + neighbourhood_group_cleansed + extra_people + 
                   review_scores_communication,data=analysisDataClean,
                 method="rf",ntree=1000,trControl=trControl,tuneGrid=tuneGrid )

#model
forest = randomForest(price ~ host_is_superhost + host_has_profile_pic + host_identity_verified + 
                        latitude + longitude + property_type + room_type + accommodates + 
                        bathrooms + bedrooms + beds + square_feet + security_deposit + 
                        cleaning_fee + guests_included + minimum_nights + availability_30 + 
                        availability_90 + availability_365 + number_of_reviews + 
                        review_scores_rating + review_scores_cleanliness + review_scores_checkin + 
                        review_scores_location + review_scores_value + is_business_travel_ready + 
                        cancellation_policy + require_guest_phone_verification + 
                        calculated_host_listings_count + reviews_per_month + host_response_time + 
                        host_listings_count + neighbourhood_group_cleansed + extra_people + 
                        review_scores_communication,data=analysisDataClean,ntree = 1000,mtry=5)

for (p in colnames(analysisDataClean)) { 
  if (class(scoringDataClean[[p]]) == "factor") { 
    levels(scoringDataClean[[p]]) <- levels(analysisDataClean[[p]]) 
  }
}
predForest = predict(forest,newdata=scoringDataClean)
rmseForest = sqrt(mean((predForest - analysisDataClean$price) ^ 2)); rmseForest

#Sumbit--------------------------------------------
submissionFile = data.frame(id = scoringData$id, price = predForest)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)
getwd()

score <-read.csv("scores.csv")
score
