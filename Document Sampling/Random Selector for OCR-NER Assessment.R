####### Selector for OCR Assessment

set.seed(1453)

# Create all 180 content pages (B-G) across 30 newspaper
pages <- expand.grid(
  newspaper = 1:30,
  page = LETTERS[2:7]
)

# 1. Randomly select one B-G page from each newspaper
first_sample <- do.call(
  rbind,
  lapply(1:30, function(d) {
    pages[pages$newspaper == d, ][sample(1:6, 1), ]
  })
)

# 2. Remove those pages from the pool
remaining <- pages[
  !paste(pages$newspaper, pages$page) %in%
    paste(first_sample$newspaper, first_sample$page),
]

# 3. Randomly select another 15 pages
second_sample <- remaining[sample(nrow(remaining), 15), ]

# Combine the two samples
sample_pages <- rbind(first_sample, second_sample)

# Sort by newspaper and page
sample_pages <- sample_pages[
  order(sample_pages$newspaper, sample_pages$page),
]

# Display the sample
sample_pages
write.csv(sample_pages, "ocr_benchmark_sample.csv", row.names = FALSE)


####### Selector for NER Assessment


set.seed(1066)

# Create all 180 content pages (B-G) across 30 newspaper
pages <- expand.grid(
  newspaper = 1:30,
  page = LETTERS[2:7]
)

# 1. Randomly select one B-G page from each newspaper
first_sample <- do.call(
  rbind,
  lapply(1:30, function(d) {
    pages[pages$newspaper == d, ][sample(1:6, 1), ]
  })
)

# 2. Remove those pages from the pool
remaining <- pages[
  !paste(pages$newspaper, pages$page) %in%
    paste(first_sample$newspaper, first_sample$page),
]

# 3. Randomly select another 15 pages
second_sample <- remaining[sample(nrow(remaining), 15), ]

# Combine the two samples
sample_pages <- rbind(first_sample, second_sample)

# Sort by newspaper and page
sample_pages <- sample_pages[
  order(sample_pages$newspaper, sample_pages$page),
]

# Display the sample
sample_pages
write.csv(sample_pages, "ner_benchmark_sample.csv", row.names = FALSE)
