from controller import Controller
from request import Request

if __name__ == '__main__':
	cont = Controller()
	req1 = Request()

	cont.queue_request(req1)

	cont.process_requests()

	cont.show()
