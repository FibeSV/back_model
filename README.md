## How to run
* Install and run Docker
* Build Docker image using `docker build . -t back_model`
* Run Docker container using `docker run --rm -it -p 80:80 back_model`
* Go to `http://127.0.0.1:80/docs` to see all available methods of the API

**All parameters are hardcoded to make the example as easy as possible** 

## Source code
* [server.py](server.py) contains API logic
* [Dockerfile](Dockerfile) describes a Docker image that is used to run the API