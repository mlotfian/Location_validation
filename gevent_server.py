from api_RF import app
from gevent.pywsgi import WSGIServer
import logging
import ssl

#http_server = WSGIServer(('0.0.0.0', 5000), app)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('C:/Certif/cert.pem','C:/Certif/key.pem')
http_server = WSGIServer(('0.0.0.0', 5000), app, ssl_context=context)
# wrap_socket = http_server.wrap_socket
# def my_wrap_socket(*args):
#     try:
#         return wrap_socket(*args)
#     except OSError as ex:
#         if ex.errno == 0:
#             # Silently kill this greenlet.
#             raise GreenletExit
# http_server.wrap_socket = my_wrap_socket
#logging.basicConfig(filename='record2.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

http_server.serve_forever()
