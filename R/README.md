# Intro to R and RStudio
Monday, August 5, 2024, 13:30 – 15:00 by Luke Maurits

## Set up

Following this hands-on session requires you to install:

* [R](https://cran.rstudio.com/) itself
* The [RStudio](https://posit.co/download/rstudio-desktop/) integrated development environment (IDE)
* The [tidyverse package](https://www.tidyverse.org/)

How you should install R and RStudio depends on your operating system, it's different across Windows, Mac and Linux.  You might have to do a bit of Googling to find instructions, but this stuff should be pretty well documented.

Once you have R and RStudio working, installing `tidyverse` will be the same no matter your OS, just run `install.packages("tidyverse")` from within an R script or the R console.

## Where to get code and data

All the code we wrote during the session now lives in
[its own Git repo](https://github.com/lmaurits/btg_2024_r_hands_on).  This published code has been polished a little bit since the session, but the overall structure and philosophy is unchanged.  I've also added a README file to explain what each file does and when you should run in each one.  You can use the Git repository to unwind my polishing and get back to the earlier raw state if that makes it easier for you to try to repeat the course based on the recording.

The raw data files are available from [the BtG NextCloud share folder](https://share.eva.mpg.de/index.php/s/gRRHDB6jGSHTytd).  The code at the Git repository expects the `data.xlsx` file to live inside a `data/` directory, so create one and save these data files there.

The Nextcloud share also contains a very roughly slide show which I didn't actually end up using on the day, but it might be a useful thing to refer to because it has some of the core ideas sketched out as dot points.  It also has my hand-drawn diagram of the overall workflow on the final slide, in case that's a thing you'd like to see again?
