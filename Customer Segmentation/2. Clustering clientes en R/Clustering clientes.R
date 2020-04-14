library(readr)
library(dendextend)
library(ggplot2)  # para dibujar los puntos iniciales
library(factoextra)   # Para hacer visualizaciones del análisis cluster
library(NbClust)   # para calcular número optimo de clusters
library(dplyr)
library(caret)
library("scatterplot3d")
library(corrplot)
library(factoextra)
library(gplots)
customer_segmentation_cat <- read_csv("C:/Users/Master/Desktop/E-3 An/Udemy/ML+Python/datasets/trabajo-final/customer-segmentation-cat2.csv")
datos <- customer_segmentation_cat
#Nos quedaremos con el 35% del DF para que el ordenador pueda ejecutar el código.
set.seed(1234)
training.ids = createDataPartition(datos$ID,p=0.35,list = F)
datos = datos[training.ids,]
datos = datos[,3:5]
#Análisis exploratorio
summary(datos)
corrplot(cor(datos), method = "color")

datnorm<-scale(datos)


dismatrix<- dist(datnorm, method = "manhattan")
#Si lo ejecutas falla
#fviz_dist(dismatrix)
#Hay que mirar el criterio óptimo
hc<- hclust(dismatrix, method = "ward.D2")
 
#Mapa de calor
heatmap.2(x = datnorm, scale = "none",
          distfun = function(x){dist(x, method = "manhattan")},
          hclustfun = function(x){hclust(x, method = "ward.D2")},
          density.info = "none",
          trace = "none",
          col = bluered(256),
          cexCol=0.8)

datos1<-datos

#Regla del codo
fviz_nbclust(x = datos, FUNcluster = kmeans, method = "wss", k.max = 15, 
             diss = get_dist(datos, method = "manhattan"), nstart = 50)
  #Regla del codo, nos indicará que el óptimo son 3 clusters, peronos interesa tener más clusters.
cluster <-  cutree(hc, k = 3) # cortamos el dendrograma para que salgan 5 clusters
datos1$cluster<-cluster   # y añadimos una columna a los datos con el cluster de cada registro
table(cluster)
#Dendrograma
plot(hc, hang=-1, main="Distancia manhattan, enlace Ward, k=5", labels=FALSE)
rect.hclust(hc, k=3, border=c("red", "blue", "green","black","orange"))
# Scatter plto 3D

colors <- c("#999999", "#E69F00", "#56B4E9")
colors <- colors[as.numeric(datos1$cluster)]
scatterplot3d(datos1, pch = 16, color=colors, angle = 225, box = TRUE)
# Cluster plot
fviz_cluster(object=list(data=datos, cluster=cluster), geom="point", ellipse=FALSE)
# Scatter plot 3D
colors <- c("#999999", "#E69F00", "#56B4E9")
colors <- colors[as.numeric(datos1$cluster)]
scatterplot3d(datos1, pch = 16,color= colors, angle = 225, box = TRUE)

# Tabla con group by por cluster.
resumen1=datos1 %>% group_by(cluster)%>% summarise_all(mean)  
resumen1$prop<-table(cluster)/ nrow(datos1)    
resumen1



