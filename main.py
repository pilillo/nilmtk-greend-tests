import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nilmtk.dataset_converters import convert_greend, convert_redd
from nilmtk import DataSet
import matplotlib as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import fhmm_exact, combinatorial_optimisation
from nilmtk.metergroup import MeterGroup
from nilmtk.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

class NILM:
	def __init__(self):
		pass
	
	def convert_dataset(self, folder, destination_file):
		#convert_greend(folder, destination_file)
		convert_redd(folder, destination_file)

	def import_dataset(self, source_file, start_end):
		self.ds = DataSet(source_file)
		
		self.ds_train = DataSet(source_file)
		self.ds_train.set_window(end=start_end)
		
		self.ds_test = DataSet(source_file)
		self.ds_test.set_window(start=start_end)
		
	def show_wiring(self, building_no):
		self.ds.buildings[building_no].elec.draw_wiring_graph()

	def show_available_devices(self, building_no):
		return self.ds.buildings[building_no].elec

	def show_available_data(self, building_no, device_id):
		return self.ds.buildings[building_no].elec[device_id].available_columns() #.device["measurements"]
		
	def get_aggregated_power(self, building_no):
		return self.ds.buildings[building_no].elec.mains().power_series_all_data() #.head()

	def get_device_power(self, building_no, device_id):
		""" 
		Returns a generator over the power timeserie
		"""
		return self.ds.buildings[building_no].elec[device_id].power_series()
		
	def get_energy_per_meter(self, building_no):
		return self.ds_train.buildings[building_no].elec.submeters().energy_per_meter().loc['active']

	def get_total_energy_per_device(self, building_no, device_id):
		return self.ds.buildings[building_no].elec[device_id].total_energy()
		
	def plot_aggregated_power(self, building_no):
		self.ds.buildings[building_no].elec.mains().plot()
		
	def plot_meter_power(self, building_no, device_id):
		self.ds.buildings[building_no].elec[device_id].plot()
		
	def plot_all_meters(self, building_no):
		self.ds.buildings[building_no].elec.plot()
	
	def plot_appliance_states(self, building_no, device_id):
		self.ds.buildings[building_no].elec[device_id].plot_power_histogram()
		
	def plot_spectrum(self, building_no, device_id):
		self.ds.buildings[building_no].elec[device_id].plot_spectrum()
		
	def plot_appliance_usage(self, building_no, device_id):
		self.ds.buildings[building_no].elec[device_id].plot_activity_histogram()
		
	def select_appliances_by_id(self, building_no, names):
		pass
		
	def select_top_consuming_appliances_for_training(self, building_no, k=5):
		return self.ds.buildings[building_no].elec.submeters().select_top_k(k)

	def select_appliances_by_type(self, t):
		import nilmtk
		meters = nilmtk.global_meter_group.select_using_appliances(type=t).all_meters()
		#print([m.total_energy() for m in meters])
		meters = sorted(meters, key=(lambda m: m.total_energy()[0]), reverse=True)   # sort by energy consumption
		#print([m.total_energy() for m in meters])
		return meters

	def create_nilm_model(self, m_type):
		if m_type is "FHMM":
			self.model = fhmm_exact.FHMM()
		elif m_type is "CombOpt":
			self.model = combinatorial_optimisation.CombinatorialOptimisation()
	
	def import_nilm_model(self, filepath, m_type):
		if m_type is "FHMM":
			self.model = fhmm_exact.FHMM()
			self.model.import_model(filepath)
		elif m_type is "CombOpt":
			self.model = combinatorial_optimisation.CombinatorialOptimisation()
			self.model.import_model(filepath)

	def train_nilm_model(self, top_devices, sample_period=None):
		if sample_period is None:
			self.model.train(top_devices)
		else:
			self.model.train(top_devices, sample_period)
	
	def save_disaggregator(self, filepath):
		self.model.export_model(filepath)
	
	def disaggregate(self, aggregate_timeserie, output_file, sample_period):
		self.model.disaggregate(aggregate_timeserie, output_file, sample_period)
		
	def plot_f_score(self, disag_filename):
		plt.figure()
		from nilmtk.metrics import f1_score
		disag = DataSet(disag_filename)
		disag_elec = disag.buildings[building].elec
		f1 = f1_score(disag_elec, test_elec)
		f1.index = disag_elec.get_labels(f1.index)
		f1.plot(kind='barh')
		plt.ylabel('appliance');
		plt.xlabel('f-score');
		plt.title(type(self.model).__name__);

#nilm = NILM()

# convert from CSV+Yaml to HDF5
#convert_greend('/home/andrea/Desktop/dataset/', '/home/andrea/Desktop/greend.h5')
#convert_redd('/home/andrea/Desktop/REDD/low_freq/', '/home/andrea/Desktop/redd.h5')

# load HDF5 dataset
#nilm.import_dataset('/home/andrea/Desktop/greend.h5')
#nilm.import_dataset('/home/andrea/Desktop/redd.h5', start_end="30-4-2011")
#nilm.import_dataset('/home/andrea/Desktop/iawe.h5')



#nilm.ds.buildings[1].elec['clothes iron'].plot_power_histogram()
#print( nilm.show_available_devices(2))				# returns the list of electric meter
#print( nilm.show_available_data(5, "dish washer"))	# returns [('power', 'active')]
#print( next(nilm.get_device_power(2, "dish washer")) )

#energy_per_meter = nilm.get_energy_per_meter(5)
#print( energy_per_meter )

#print(nilm.get_total_energy_per_device(5, "fridge"))

#dw = nilm.ds.buildings[2].elec['dish washer']
#print dw.available_columns()
#dw.plot()
#print( dir(dw) ) #.available_columns() )
#print( type(dw) )
#print(nilm.ds.buildings[3].elec.mains().plot())
#print(dw.plot_power_histogram())
#dw.plot_power_histogram()
#nilm.plot_all_meters(1)
#nilm.plot_meter_power(6, "fridge")
#nilm.plot_aggregated_power(2)

#s =nilm.select_appliances_by_type("fridge")	# get all fridges in the dataset ordered by energy DESC

#print(nilm.ds.buildings[6].elec.submeters().all_meters )

def create_group():
	nilm.create_nilm_model("FHMM")#"CombOpt")
	device_family = []
	device_family.append( nilm.select_appliances_by_type("fridge")[0] )
	device_family.append( nilm.select_appliances_by_type("washing machine")[0] )
	device_family.append( nilm.select_appliances_by_type("dish washer")[0] )
	device_family.append( nilm.select_appliances_by_type("light")[0] )
	device_family.append( nilm.select_appliances_by_type("washer dryer")[0] )
	device_family.append( nilm.select_appliances_by_type("electric space heater")[0] )

	#top_devs = nilm.select_top_consuming_appliances_for_training(6, 5)
	print device_family
	return MeterGroup(device_family), device_family
	
def train_group(group):
	nilm.train_nilm_model(group, sample_period=60)




# Example at https://github.com/nilmtk/nilmtk/blob/master/docs/manual/user_guide/disaggregation_and_metrics.ipynb

train = DataSet('/home/andrea/Desktop/redd.h5')
test = DataSet('/home/andrea/Desktop/redd.h5')

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")

train_elect = train.buildings[1].elec
test_elec = test.buildings[1].elec
best_devices = test_elec.submeters().select_top_k(k=5)

test_elec.mains().plot()

fhmm = fhmm_exact.FHMM()
fhmm.train(best_devices, sample_period=60)

# Save disaggregation to external dataset
#output = HDFDataStore('/home/andrea/Desktop/nilmtk_tests/redd.disag-fhmm.h5', 'w')
"""
fhmm.disaggregate(test_elec.mains(), output, sample_period=60)
output.close()

# read result from external file
disag_fhmm = DataSet(output)
disag_fhmm_elec = disag_fhmm.buildings[building].elec

disagg_fhmm.plot()
"""
"""
from nilmtk.metrics import f1_score
f1_fhmm = f1_score(disag_fhmm_elec, test_elec)
f1_fhmm.index = disag_fhmm_elec.get_labels(f1_fhmm.index)
f1_fhmm.plot(kind='barh')
plt.ylabel('appliance');
plt.xlabel('f-score');
plt.title("FHMM");
"""
plt.show()

