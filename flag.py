pl.when(pl.col("date1").is_null() | pl.col("date2").is_null()).then(None)
  .when(pl.col("date2") < pl.col("date1").dt.offset_by("24mo")).then(1)
  .otherwise(0).cast(pl.Int8).alias("flag_24m")
