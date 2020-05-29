import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._


val filename = "C:/spark/loudacre_BigData/loudacre_Dataset"
val loudacre_data = sc.textFile(filename)

val loudacre_data_split = loudacre_data.map(x => x.split(",")).map(s => (s(3),s(4))).toDF

loudacre_data_split.show(10)

loudacre_data_split.filter($"_1"==="0").show(10)

loudacre_data_split.filter($"_1" === "0").count

val la_df = loudacre_data_split.filter($"_1" =!= "0")

la_df.filter($"_1" === "0").count

val la_rdd = la_df.rdd.map(row => List(row.getString(0),row.getString(1)))

val vectors = la_rdd.map(s => Vectors.dense(s(0).toDouble,s(1).toDouble)).cache()

val numClusters = 3
val numIterations = 20

val kmeansmodel = KMeans.train(vectors,numClusters,numIterations)

kmeansmodel.clusterCenters.foreach(println)

 kmeansmodel.computeCost(vectors)

kmeansmodel.k

val kmeansModel_BC = sc.broadcast(kmeansmodel)

val cluster_df = kmeansModel_BC.value.predict(vectors).toDF

cluster_df.show

val df1 = cluster_df.withColumn("id",monotonically_increasing_id())
val df2 = la_df.withColumn("id",monotonically_increasing_id())

df1.show(5)

df2.show(5)

val df3 = df2.join(df1,"id")
df3.show

df3.groupBy("value").count().show()

val newNames = Seq("latitude","longitude","cluster")

val finaldf = df3.drop("id").toDF(newNames:_*)
finaldf.show

 finaldf.filter($"cluster" === "0").show

 finaldf.filter($"cluster" === "1").show

 finaldf.filter($"cluster" === "2").show