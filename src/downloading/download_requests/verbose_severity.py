
from enum import Enum
from functools import total_ordering


@total_ordering
class VerboseSeverity(Enum):
	NONE  = 0
	ERROR = 1
	WARN  = 2
	DEBUG = 3
	INFO  = 4

	def __lt__(self, other):
		if self.__class__ is other.__class__:
			return self.value < other.value
		return NotImplemented


	def __eq__(self, other):
		if self.__class__ is not other.__class__:
			return self.value == other.value
		return NotImplemented


	def __ne__(self, other):
		return not self == other


	def __le__(self, other):
		return self == other or self < other


	def __gt__(self, other):
		return not self <= other


	def __ge__(self, other):
		return not self < other

