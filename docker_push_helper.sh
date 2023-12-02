docker build -f DependencyDockerfile -t dependency-image .
docker build -f FinalDockerfile -t final-image .
docker run final-image