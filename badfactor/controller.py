from drawer import Drawer
from request import Request


class Controller:
	def __init__(self):
		self._requests_queue = []
		self._drawer = None
		self._calculator = None


	# queue up each chart to put in figure
	def queue_request(self, req: Request):
		self._requests_queue.append(req)

	# process request queue & show graph, only use when queue is populated for a full graph
	def process_requests(self):
		self._drawer = Drawer(len(self._requests_queue))

		for i, req in enumerate(self._requests_queue):
			self._process_request(req, i)

		self._drawer.show()


	def _process_request(self, req: Request, axes_index: int):
		self._calculator.load_values(req)

		if self._calculator.values is None:
			self._calculator.calculate_values(req)

		# self._drawer.draw(req, self.values, axes_index)
		self._write_values(req)


	def _write_values(self, req: Request):
		pass

