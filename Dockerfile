FROM python:3.9.16
LABEL maintainer="wangjunhong <wangjunhong@email.ncu.edu.cn>"
COPY . /xss-model/
COPY pip.conf /etc/pip.conf
WORKDIR /xss-model
RUN ["pip", "install" ,"-r", "requirements.txt"]
CMD ["python", "XssModelApplication.py"]