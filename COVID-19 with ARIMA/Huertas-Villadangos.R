datoss = read.csv("Huertas-Villadangos.csv", header=T, sep=",") #leemos los datos
datoss=datoss[,-c(1:4)] #eliminamos las cuatro primeras columnas, que hacen referencia al pais/region y sus coordenadas
total=colSums(datoss)
datoss=rbind(datoss,total) #sumamos todas as filas de cada columna y añadimos el resultado en la ultima fila
datos=datoss[-c(1:(nrow(datoss)-1)),] #nos quedamos con esa fila que contiene los contagios diarios acumulados a nivel mundial
for (col in c(ncol(datos):2)){
  datos[,col]=datos[,col]- datos[,col-1]
} #restamos cada dia a su anterior, para obtener los contagios a nivel mundial de cada dia concreto
datos=t(datos) #trasponemos los datos

ST = ts(datos, start = c(2020,22)  , frequency = 365) #convertimos los datos en un objeto de serie temporal
ST
plot(ST, xlab = "Tiempo", ylab = "Casos confirmados", col='blue') #pintamos la serie

library("TTR")
Tendencia <- SMA(ST, n=20)
plot(Tendencia)

#pintamos la serie, sus FAS y sus FAP
par(mfrow=c(1,3))
plot(ST,main="ST",col='blue')
acf(ST, main="FAS")
pacf(ST,main="FAP")

#identificamos los atipicos
library(tsoutliers)
outliers <- tso(ST)
outliers
plot(outliers)

#elegimos como conjunto de validacion los ultimos 20 dias de los que tenemos datos
nValidacion = 25
nEntrenamiento = length(ST)-nValidacion #el resto de datos forman parte del conjunto de entrenamiento

Train.data = window(ST, start = c(2020, 22), end = c(2020, 22+nEntrenamiento)) #extraemos los datos del conjunto de entrenamiento 
Train.data

#vemos si es necesario aplicar transformacion Box-Cox
library(forecast)
Lambda=BoxCox.lambda(Train.data)
Lambda
#como el lambda es 1, significa que no hay heterocedasticidad, por lo que no es necesario aplicar transformacion Box-Cox

#probamos a diferenciar manualmente, para ver cuantas diferencias son necesarias para hacer la serie estacionaria
Btrain <- diff (Train.data, differences = 1)
ggtsdisplay(Btrain, lag.max = 50)

B2train <- diff (Btrain, differences = 1)
ggtsdisplay(B2train, lag.max = 50)


#ajustamos un modelo ARIMA con auto.arima
auto.arima(Train.data,seasonal=FALSE)

auto.arima(Train.data, stepwise=FALSE,approx=FALSE,seasonal=FALSE, trace=TRUE)


#ajustamos un modelo ARIMA con un bucle for para ver si conseguimos mejorar el resultado de auto.arima

p=0
d=1
q=0
ordenregular = 2 #Vamos a probar hasta 2 diferenciaciones, porque son las que hemos visto que eran necesaria cuando antes hemos diferenciado manualmente

#Partimos de un modelo naive
mejor.modelo = Arima(Train.data, order=c(0,0,0),
                     include.constant = TRUE)
mejor.modelo$aic

for(p in 0:ordenregular)
{
  for(d in 1:ordenregular) 
  {
    for(q in 0:ordenregular)
       {
        modelo= Arima(Train.data, order=c(p,d,q),
                      include.constant = TRUE)
            
            pval=Box.test(modelo$residuals,lag=10,type="Lj")#El test de Ljung-Box (portmanteau test) tambi?n se puede hacer as?
            sse=sum(modelo$residuals^2)
            
            if(modelo$aic < mejor.modelo$aic)
            {
              mejor.modelo=modelo  
            }
            cat(p,d,q,"AIC=",modelo$aic,"SSE=",sse,"p-valor=",pval$p.value,"\n")
    }
  }
}
mejor.modelo

#en ambos caso, el modelo que tiene menor AIC es un ARIMA(1,2,2), pero al analizar 
#los coeficientes, descubrimos que el ar1 no es significativo, por lo que decidimos ajustar
#manualmente un ARIMA(0,2,2) porque la diferencia de AIC es muy pequeña
#ajustamos el modelo manualmente
arima.fit.entrenamiento = Arima(Train.data, order=c(0,2,2),
                                include.constant = TRUE)
arima.fit.entrenamiento

library(lmtest)  
coeftest(arima.fit.entrenamiento) #vemos si los coeficientes con significativos

#hacemos el diagnostico de los residuos

tsdiag(arima.fit.entrenamiento) #pasa el test de Ljung-Box sin problemas, y el resto parece bien

#hacemos la prediccion para los 20 dias del conjunto de validacion y para los 30 dias siguiente, en total 25 dias
prediccion = forecast(arima.fit.entrenamiento, h=55)
prediccion

#pintamos la prediccion
plot(prediccion)
lines(ST)

plot(prediccion, ylab = "Casos confirmados en todo el mundo", xlab = "Tiempo")
lines(arima.fit.entrenamiento$fitted, lwd=2, col='green', lty=1)
lines(ST,col= "red")

#probamos un modelo Holt y comparamos los resultados
Holt=HoltWinters(Train.data, gamma=FALSE)
Holt
plot(Holt)
Prediccion = predict(Holt, n.ahead = 55, prediction.interval = T, level = 0.95) 
plot(Holt, Prediccion)
lines(ST,col= "black")

#utilizamos el paquete prophet para hacer otra prediccion 
install.packages("prophet")
library(prophet)

#hacemos la prediccion con el conjunto de entrenamiento
df_prophet=data.frame(seq(as.Date("2020-01-22"), as.Date("2020-03-29"), "days"),Train.data)
names(df_prophet)=c("ds","y")
m=prophet(df_prophet)
future <- make_future_dataframe(m, periods = 55)
forecast <- predict(m, future)

#repetimos lo mismo pero con todos los datos, para poder incluirlos en el plot siguiente
df_prophet2=data.frame(seq(as.Date("2020-01-22"), as.Date("2020-04-22"), "days"),ST)
names(df_prophet2)=c("ds","y")
m2=prophet(df_prophet2)

plot(m2,forecast)
prophet_plot_components(m2, forecast)
dyplot.prophet(m2, forecast)

citation()
citation(package="TTR")
citation(package="tsoutliers")
citation(package="forecast")
citation(package="lmtest")
citation(package="prophet")

#Modelo ARIMA(0,2,2) con todos los datos
Lambda=BoxCox.lambda(ST)
auto.arima(ST, stepwise=FALSE,approx=FALSE,seasonal=FALSE, trace=TRUE)
arima.fit.entrenamiento = Arima(ST, order=c(0,2,2),
                                include.constant = TRUE)
coeftest(arima.fit.entrenamiento)
coeftest(arima.fit.entrenamiento)
prediccion = forecast(arima.fit.entrenamiento, h=30)
plot(prediccion, ylab = "Casos confirmados en todo el mundo", xlab = "Tiempo")
