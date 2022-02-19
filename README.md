# SSML_BD_006_089_126_282

 This repository hosts the code for SSML_BD_006_089_126_282, the Spark Machine Learning project for the course Big Data (UE19CS322) at PES University. The project involves conducting Sentiment Analysis via Online Machine Learning, using Spark Streaming. 

## Committing

 We use a rebase-oriented workflow. We do not use merge commits. This means to get your local branch up-to-date with the upstream, you would use
  ```
  git pull --rebase upstream main
  ```
 instead of regular `git pull`. It’s best to write your commits prefacing the file you changed, but if you don’t, you can always fix your history using `git rebase -i`. An example of a good commit would be
  ```
  model: Add classifier class.
  ```

## Executing
 Install all requirements as follows:
  ```
  pip3 install -r requirements.txt
  ```
 Now install the nltk corpus, by doing the following:
  1. Open Python in your shell:
        ```
        python3
        ```
  2. Run the following:
        ```
        import nltk
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("stopwords")
        ```
 Run 'stream.py' to start the Data Stream, as follows:
  ```
  python3 stream.py -f folder-name-with-train-and-test-files -b <batch-size>
  ```

 To try out the project itself, run
  ```
  /opt/spark/bin/spark-submit train.py > train.log
  ```

Contributors
-----
 - [Aanchal Narendran](https://github.com/aanchal-n)
 - [Arushi Kumar](https://github.com/arushi32001)
 - [Chethas K](https://github.com/Chethas47)
 - [Murali Krishna](https://github.com/LaRuim)

----

 This is an automatically generated README by murl.AI <br/>
 All rights reserved.