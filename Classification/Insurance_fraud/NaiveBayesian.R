data1.raw<-read.csv("data_1.csv",stringsAsFactors=FALSE) #파일을 불러온다.
str(data1.raw) #파일의 구조를 파악한다.
data1.new<-subset(data1.raw,DIVIDED_SET==1) #DIVIDED_SET=1이 우리가 사용할 학습파일이다.
str(data1.new) #파일의 구조를 파악한다.

dt1 <- data.frame(data1.new$SEX, data1.new$AGE, data1.new$RESI_COST,
                  data1.new$RESI_TYPE_CODE,data1.new$TOTALPREM,data1.new$SIU_CUST_YN, stringsAsFactors=FALSE) #원하는 열만 선택한다.
names(dt1) <- c("성별","연령","주택가격","거주형태","총납입료합계","사기여부") #만들어진 열의 이름을 바꾸어준다.
str(dt1) #파일의 구조를 파악한다.

dt2 <- na.omit(dt1) #결측치를 제거한다. 여기서 결측치를 포함한 것이 전체수에 비해 적기 때문에 그냥 시행했다.
str(dt2) #파일구조를 파악한다.

# write.csv(dt2, file="data1.new.csv",row.names=FALSE) 만든 데이터를 csv로 저장한다.

library("psych") #pairs.panels 를 포함하고 있다.
library("e1071") #나이브베이즈를 포함하고 있다.
library("caret") #ConfusionMatrix를 포함하고 있다.

dt3 <- dt2

sex <- dt3$성별
age <- dt3$연령
cost <- dt3$주택가격
type <- dt3$거주형태
total <- dt3$총납입료합계
fake <- dt3$사기여부

dt3 = data.frame(sex, age, cost, type, total, fake)
set.seed(2000)
dt3.size <- length(fake)
dt3.train.size <- round(dt3.size*0.7) # 70% for training
dt3.validation.size <- dt3.size - dt3.train.size # The rest for testing
dt3.train.idx <- sample(seq(1:dt3.size),dt3.train.size) # indeces for training
dt3.train.sample <- dt3[dt3.train.idx,]
dt3.validation.sample <- dt3[-dt3.train.idx,]

classf <- naiveBayes(
  subset(dt3.train.sample, select = -fake),
  dt3.train.sample$fake, laplace=1)
classf
preds <- predict(classf, subset(dt3.validation.sample, select = -fake))
table(preds, dt3.validation.sample$fake)
round(sum(preds == dt3.validation.sample$fake, na.rm=TRUE) / length(dt3.validation.sample$fake), digits = 2) #accuracy

confusionMatrix(table(preds, dt3.validation.sample$fake))

