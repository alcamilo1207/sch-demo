import os
import json
import math
import pickle
import requests
import numpy as np
import random as rd
import pandas as pd
from datetime import datetime, timedelta

metadata = {
    "region" : "Germany/Luxembourg [€/MWh] Original resolutions",
    "energy prices filename" : "Day-ahead_prices_2019_2024_Hour.csv",
    "pv col name" : "Photovoltaics [MWh] Original resolutions",
    "pv cap col name" : "Photovoltaics [MW] Original resolutions",
    "pv data filename" : "Actual_generation_2019_2024_Quarterhour_DE.csv",
    "installed capacity filename" : "Installed_generation_capacity_2019_2024_DE.csv",
}

path = os.getcwd()

try:
    pv_data = pd.read_csv(os.path.join(path, 'energy_module', metadata["pv data filename"]), sep=";", low_memory=False)
    eprices_data = pd.read_csv(os.path.join(path, 'energy_module', metadata["energy prices filename"]), sep=";", low_memory=False)
    installed_capacity_data = pd.read_csv(os.path.join(path, 'energy_module', metadata["installed capacity filename"]), sep=";", low_memory=False)
except OSError as e:
    pv_data = pd.read_csv(os.path.join(path, 'integration','energy_module', metadata["pv data filename"]), sep=";", low_memory=False)
    eprices_data = pd.read_csv(os.path.join(path, 'integration', 'energy_module', metadata["energy prices filename"]), sep=";", low_memory=False)
    installed_capacity_data = pd.read_csv(os.path.join(path, 'integration', 'energy_module', metadata["installed capacity filename"]), sep=";", low_memory=False)
except Exception as e:
    print("Unexpected error", e)

class Job:
    def __init__(self, id: int, size, start_timestamp = None, finish_timestamp = None, actual_duration = 0, completed=False):
        self.id = id
        self.size = size
        self.start_timestamp = start_timestamp
        self.finish_timestamp = finish_timestamp
        self.actual_duration = actual_duration
        self.completed = completed
        self.hold = False

    def __str__(self):
            return f"Job {self.id}, size: {self.size}, duration: {self.actual_duration}, completed: {self.completed}"


class Machine:
    def __init__(self,id: int, speed = 1.0, energy_usage = 1.0):
        self.id = id
        self.speed = speed
        self.energy_usage = energy_usage
        self.time = 0
        self.history = []

    def __str__(self):
        return f"Machine {self.id}, speed: {self.speed}, energy_usage: {self.energy_usage}"


class Scheduler:
    """
    Generates daily schedules with synthetic data.
    -> Time resolution: 1 minute
    """
    def __init__(self, n_machines = 3):
        self.n_jobs = 0
        self.n_machines = 0
        self.jobs = []
        self.machines = []

        default_eu = [0.75, 1, 1.25]
        for i in range(n_machines):
            self.n_machines += 1
            self.machines.append(Machine(id = self.n_machines-1, speed = 132, energy_usage = default_eu[i]))

    def __str__(self):
        return f"jobs {self.n_jobs}, Machines: {self.n_machines}"
    

    def load(self,filename="scheduler"):
        with open(f'{filename}.pkl', 'rb') as file:
            loaded_object = pickle.load(file)

        print(">>> Scheduler has been loaded.")
        return loaded_object

    def save(self,filename="scheduler"):
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self, file)

        print(">>> Scheduler has been saved.")

    def create_job(self):
        size = rd.uniform(100,300)
        self.n_jobs += 1
        self.jobs.append(Job(self.n_jobs, size = size))

    def create_machine(self):
        speed = rd.uniform(0.75,1.25)
        energy_usage = rd.uniform(0.5,1.5)
        self.n_machines += 1
        self.machines.append(Machine(self.n_machines-1,speed,energy_usage))

    def get_pending_jobs(self):
        jobs = self.jobs
        pending_jobs = []
        for j in jobs:
            if j.completed == False and j.hold == False:
                pending_jobs.append(j)
        
        return pending_jobs

    def convert_timestamp_to_minutes(self, timestamp, duration):
        start = timestamp.hour * 60 + timestamp.minute
        end = start + duration
        return [start, end]

    def convert_minutes_to_timestamp(self,minutes):
        # Define the start date
        start_date = datetime(2024, 1, 1)

        # Create a timedelta object for the given minutes
        time_delta = timedelta(minutes=minutes)

        # Calculate the new timestamp by adding the timedelta to the start date
        new_timestamp = start_date + time_delta

        # Return the new timestamp
        return new_timestamp

    def get_duration(self,job, machine):
        size = job.size
        speed = machine.speed
        return math.ceil(size/speed)
    
    def get_total_energy_usage(self,job, machine):
        size = job.size
        energy_usage = machine.energy_usage
        return size*energy_usage

    def get_schedule(self):
        schedule = []
        sch_df_cols = ["assigned_to", "energy", "name", "size", "start_timestamp", "finish_timestamp", "actual_duration"]
        machines = self.machines
        for i, m in enumerate(machines):
            for j in m.history:
                schedule.append([f"machine {i}",
                                 self.get_total_energy_usage(j,m),
                                 f"job {j.id}",
                                 j.size,
                                 j.start_timestamp,
                                 j.finish_timestamp,
                                 j.actual_duration
                             ]
                        )

        sch_df = pd.DataFrame(schedule,columns=sch_df_cols)
        return sch_df.sort_values(by=['start_timestamp'], ascending=[True], ignore_index=True)
    
    def _get_schedule(self, schedule, date):
        new_schedule = []
        start = datetime(date.year, date.month, date.day, 0, 0)
        machine_times = [0,0,0]
        cols = ["assigned_to", "energy", "name", "size", "actual_duration","reward", "start_timestamp", "end_timestamp"]
        for i, item in enumerate(schedule):
            assigned_to = item[0]
            actual_duration = item[3]

            if item[2] > 0:
                name = f"job {i+1}"
            else:
                name = "job 0"

            start_timestamp = start + timedelta(minutes = machine_times[assigned_to])
            end_timestamp = start_timestamp + timedelta(minutes = actual_duration)
            machine_times[assigned_to] += actual_duration

            m_label = f"machine {assigned_to}"
            new_schedule.append([m_label, item[1], name , item[2], actual_duration, item[4], start_timestamp, end_timestamp])
        
        return pd.DataFrame(new_schedule, columns=cols)

    def get_power(self):
        schedule = self.get_schedule()
        #schedule["start_timestamp"] = pd.to_datetime(schedule["start_timestamp"])
        power_schedules = []
        for i, m in enumerate(self.machines):
            m_label = f"machine {i}"
            m_schedule = schedule[schedule["assigned_to"] == m_label]
            idle_times = m_schedule[m_schedule["name"] == "job 0"]
            it_int = [self.convert_timestamp_to_minutes(Helper().int_to_time(start,60), duration) for start, duration in
                      zip(idle_times["start_timestamp"], idle_times["actual_duration"])]
            m_power = m.speed * m.energy_usage
            m_pw_sch = np.full(1440, m_power)
            if len(it_int) > 0:
                for it in it_int:
                    start, end = it[0], it[1]
                    m_pw_sch[start:end] = 0.0

            power_schedules.append(m_pw_sch)

        total_power = np.copy(power_schedules[0])
        for i in range(1, len(power_schedules)):
            total_power += power_schedules[i]

        power_schedules.append(total_power)
        start_time = schedule["start_timestamp"][0]
        timestamps = pd.date_range(start_time,periods=1440,freq='1min')
        cols = [f'Machine {i}' for i in range(np.shape(power_schedules)[0]-1)]
        cols.append('Total power')

        df = pd.DataFrame(np.transpose(power_schedules),index=timestamps,columns=cols)
        
        return df
    
    def _get_power(self, schedule, date):
        machine_speeds = [132,132,132] # kg/h
        machine_eu = [0.75, 1, 1.25] # kWh/kg
        np_schedule = np.array(schedule)
        machines = np.unique(np_schedule[:,0])
        cols = [f"machine {i}" for i in range(len(machines))]
        cols.append("Total power")

        power_data = np.zeros((96, len(machines) + 1))

        timestamps = pd.date_range(date, periods=96, freq="15min")

        df = pd.DataFrame(power_data,index=timestamps, columns = cols)

        for item in np_schedule:
            id = int(item[0][-1])
            energy = item[1]
            actual_duration = int(item[4]/15)
            start_timestamp = item[6]
            if energy == 0:
                df.at[start_timestamp, machines[id]] = 0
            else:
                for i in range(actual_duration):
                    df.at[start_timestamp+timedelta(minutes=i*15), machines[id]] = machine_speeds[id]*machine_eu[id]

        if len(df) > 96:
            df = df.drop(index=date+timedelta(days=1))

        for i, m in enumerate(machines):
            df["Total power"] += df[m]

        return df

    def schedule_job(self, job, machine, idle_prob = 0.1):
        start_timestamp = datetime(year = 2023, month = 1, day = 1) + timedelta(minutes = machine.time)
        if rd.random() > idle_prob:
            duration = self.get_duration(job, machine)
            
            machine.time += duration

            job.start_timestamp = Helper().time_to_int(start_timestamp,60)
            job.finish_timestamp = Helper().time_to_int(start_timestamp + timedelta(minutes=duration),60)
            job.actual_duration = duration
            job.completed = True
            machine.history.append(job)

        else:  # Idling job
            duration = math.ceil(rd.uniform(50,300))
            if machine.time + duration < 1440:
                machine.time += duration
            else:
                duration = 1440 - machine.time
                machine.time += duration
            
            finish_timestamp = start_timestamp + timedelta(minutes=duration)
            machine.history.append(Job(id = 0,
                                       size = 0.0, 
                                       start_timestamp=Helper().time_to_int(start_timestamp,60), 
                                       finish_timestamp = Helper().time_to_int(finish_timestamp,60), 
                                       actual_duration = duration, 
                                       completed = True
                                    )
                                )

    # Very important method
    def create_schedule(self, method=None, idle_prob = 0.15):
        """
        Generate schedule for all machines using synthetic data
        """
        print(">>> Creating schedule...")
        if method == "random":
            print(f">>> Generating schedule using \"{method}\" order.")
        else:
            print(f">>> Generating schedule using \"sequential\" order.")
  
        machines = self.machines

        pending_jobs = self.get_pending_jobs()
        while len(pending_jobs) > 0:  # Loop until the scheduling end condition is met
            for j in pending_jobs: # Loop through each pending job
                # job selection method
                if method == "priority":
                    pass

                elif method == "random":
                    pick = rd.randint(0,len(pending_jobs)-1)
                    j = pending_jobs[pick]
                else:
                    pass

                times = [m.time for m in machines]
                
                # look for a machine for the specific job
                for m in machines:  
                    # Check if the machine is available
                    if m.time == 0 or m.time == min(times):
                        # Check if the work can be done before the end of the day
                        if m.time + self.get_duration(j, m) < 1440:
                            self.schedule_job(j, m, idle_prob)
                            break
                        else:
                            # Check for another that can accomplish the work before the end of the day
                            for m in machines:
                                if m.time + self.get_duration(j, m) < 1440:
                                    self.schedule_job(j, m, idle_prob)
                                    break

                            j.hold = True

                pending_jobs = self.get_pending_jobs()
                break

    def main(self, n_jobs, n_machines, choice, method=None, idle_prob = 0.15):
        ## Load or create an schedule from scratch
        if choice == "load":
            self.load()

        elif choice == "create":
        
            for _ in range(n_jobs): self.create_job() # Create job with radom parameters

            for _ in range(n_machines): self.create_machine() # Create machine with radom parameters
        
            print(">>> Jobs and machines generated with random paremeters!")
            return self
        else:
            print("Invalid option")
            return None

        self.create_schedule(method, idle_prob) # Create schedule and store it            


class EnergyPrices:
    def __init__(self, region = metadata["region"]):
        self.region = region

    def get_price_at_minute(self, day_prices, minute):
        if minute < len(day_prices)*60:
            h = int(minute/60)
        elif minute == len(day_prices)*60:
            h = len(day_prices)-1
        else:
            h = len(day_prices)-1
            print(f"minute greater than {len(day_prices)*60}!")
            
        price = day_prices[h]
        return price

    def get_hour_at_minute(self, minute):
        h = int(minute/60)
        return h

    def time_to_int(self, time, _timestep_size_seconds) -> int:
        """Converts `time` from pandas.Timestamp / pandas.Timedelta into the internal integer format - uses the globally set time unit (e.g. minutes / seconds)
    
        Args:
            time (_type_): Timestamp / Timedelta to convert
    
        Returns:
            int: Converted time
        """
        assert not (isinstance(time, int) or isinstance(time, np.integer)), "time_to_int: is already integer"
    
        if isinstance(time, pd.Timestamp):
            total_seconds = time.to_pydatetime().timestamp() # seconds since start of Unix epoch
        elif isinstance(time, pd.Timedelta):
            total_seconds = time.total_seconds()
        else:
            raise ValueError(f"time_to_int: unknown input format, type: {type(time)}")
        return int(total_seconds // _timestep_size_seconds) # somehow this results in a float when omitting int(...)? 


    def int_to_time(self, time: int, _timestep_size_seconds) -> pd.Timestamp:
        """Converts `time` from the internal integer format to pandas.Timestamp - uses the globally set time unit (e.g. minutes / seconds)
    
        Args:
            time (int): Timestamp to convert
    
        Returns:
            pd.Timestamp: Converted time
        """
        if isinstance(time, int) or isinstance(time, np.integer):
            return pd.Timestamp.fromtimestamp(time * _timestep_size_seconds)
        else:
            raise ValueError(f"time_to_int: already in time format / unknown input format? type: {type(time)}")

    def create_empty_energy_prices_df(self, start_datetime):
        date_range_start = pd.date_range(start_datetime.strftime("%Y/%m/%d"), periods=24, freq='h')
        date_range_end = pd.date_range(start=date_range_start[1], periods=24, freq='h')
        df = pd.DataFrame(np.full((24,3),0.0),columns=['Start date','End date', self.region])
        df['Start date'] = date_range_start
        df['End date'] = date_range_end
        return df
    
    def get_historic_day_prices(self, year=2019, month=1, day=1, hour=0):
        """
        Get a list with the energy prices data at 1-hour resolution for a given day
        """
        prices = []
        date = datetime(year, month, day,hour)
        os_type = os.name
        df = eprices_data
        for i in range(24):
            ts = date + timedelta(hours=i)

            # Change the format based on operative system
            if os_type == 'posix':
                date_str = ts.strftime("%b %-d, %Y %-I:%M %p")
            else:
                date_str = ts.strftime("%b %#d, %Y %#I:%M %p")

            row = df[df["Start date"] == date_str]
            row_value = row[metadata["region"]].values[0]

            # Remove commas from the string
            #row_value = row_value.replace(',', '')

            try:
                price_value = float(row_value)
            except:
                price_value = 0.0

            prices.append(price_value)

        return prices

    def get_prices(self, date):
        """"
        Request energy marked data through the www.smard.de API.
        """

        # Define the URL to send the request to
        url = 'https://www.smard.de/nip-download-manager/nip/download/market-data'  # Replace with the actual URL

        # Define the headers for the request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*'
        }

        # Get the current day datetime starting at 00:00 hours
        picked_day = datetime(date.year, date.month, date.day)

        # Get the next day datetime starting at 00:00 hours (2 hours shift corrected)
        start_datetime = picked_day #+ timedelta(days=1) #- timedelta(hours=2)
        print("start_datetime",start_datetime)

        # Get the other next day datetime starting at 00:00 hours
        end_datetime = start_datetime + timedelta(days=1)

        # Converting datetime to integer format
        start_date = int(start_datetime.timestamp()*1000)
        end_date = int(end_datetime.timestamp()*1000)

        # Define the data to be sent in the request
        data = {"request_form":[{"format":"CSV","moduleIds":[8004169,8004170,8000251,8005078,8000252,8000253,8000254,8000255,8000256,8000257,8000258,8000259,8000260,8000261,8000262,8004996,8004997],"region":"DE","timestamp_from":start_date,"timestamp_to":end_date,"type":"discrete","language":"en","resolution":"hour"}]}

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        # Send the POST request with the JSON data
        response = requests.post(url, headers=headers, data=json_data)

        # Check the response status code
        if response.status_code == 200:
            # Successful request
                
            # Decode the response content using utf-8-sig
            csv_str = response.content.decode("utf-8")
            list = [x.split(';') for x in csv_str.split('\n')]
            list[0][0] = 'Start date'
            # Convert string to data frame
            df = pd.DataFrame(list[1:-1])

            # Check if dataset is empty
            if df.iloc[0,0] == 'No data for submitted query\r':
                return self.create_empty_energy_prices_df(start_datetime)

            # Adding the headings
            df.columns = list[0][:df.shape[1]]

            return df
        else:
            # Unsuccessful request
            print(f"Request failed with status code: {response.status_code}")
            print("Response content:", response.text)
            return self.create_empty_energy_prices_df(start_datetime)

    def calc_cost(self, prices, scheduler):
        """
        Calculate the energy cost of a given machine.
        """
        cost = 0 # Euros
        for m in scheduler.machines:
            speed = m.speed # kg/min
            energy_usage = m.energy_usage # kWh/kg
            power = 60 * speed * energy_usage # kW

            # Loop through all the jobs completed by the machine
            for job in m.history:
                start_timestamp = self.int_to_time(job.start_timestamp, 60)
                duration_in_mins = job.actual_duration

                # Calculate the start time in minutes betwenn the 0:00 hours and the job time stamp
                date = datetime(start_timestamp.year, start_timestamp.month, start_timestamp.day)
                start_timestamp_in_mins = int((start_timestamp - date).total_seconds() / 60)

                #prices = self.get_historic_day_prices(year, month, day)
                for i in range(duration_in_mins):
                    p = self.get_price_at_minute(prices, start_timestamp_in_mins + i) / 1000 # Euros/kWh
                    cost += p * power * 1 / 60

        return cost

    def metrics(self, prices_df, scheduler):
        """
        This method calculates the energy cost for a given sythetic job schedule. 
        This is only used for the web app.
        """
        # Start all the output variables at zero to avoid unassigned variables
        performance, total_cost, total_energy, total_production, total_number_of_jobs = 0,0,0,0,0 
        
        # Getting variables for internal operations
        #power_values = power_schedule.values/(60*1000)
        job_schedule = scheduler.get_schedule()
        energies = job_schedule['energy'].values
        sizes = job_schedule['size'].values

        # Counting number of jobs
        for size in sizes:
            if size > 0:
                total_number_of_jobs += 1

        total_production += sizes.sum()
        total_energy += energies.sum()

        # First check if the dataframe has the right shape
        if prices_df.shape == (24,19):
            prices_str = prices_df[self.region].values
            try:
                # Convert prices array values from strings to floats
                prices = [float(item) for item in prices_str]
                #total_cost += np.sum([pow[0]*self.get_price_at_minute(prices,min) for min, pow in enumerate(power_values)])
                total_cost += self.calc_cost(prices, scheduler)
                performance += total_production/(total_energy*total_cost)
            except Exception as e:
                if str(e) == "could not convert string to float: '-'":
                    print("Day-ahead prices not available")
                else:
                    print(e)

        return performance, total_cost, total_energy, total_production, total_number_of_jobs
    
    def _metrics(self, schedule, prices, pw_df):
        total_cost, total_energy, total_production, total_number_of_jobs = 0,0,0,0
        totals = pw_df.sum()
        total_energy = totals["Total power"]/4

        for item in schedule:
            size = item[2]
            if size > 0:
                total_number_of_jobs += 1
                total_production += size

        for i, power in enumerate(pw_df["Total power"]):
            price = prices[int(i/60)]/1000
            total_cost += power*price*(15/60)

        return total_cost, total_energy, total_production, total_number_of_jobs
    
    def prices_profile(self, prices_df):
        price_profile = np.ones((96,))
        if prices_df.shape == (24,19):
            prices_str = prices_df[self.region].values
            try:
                # Convert prices array values from strings to floats
                prices = np.array([float(item) for item in prices_str])
                average = prices.mean()
                max_lim = prices.max()
                threshold = (max_lim-average)*0.25 + average
                for i, price in enumerate(prices):
                    if price > threshold:
                        price_profile[i] = 2
                    # else:
                    #     price_profile.extend([1,1,1,1])

                return price_profile, prices

            except Exception as e:
                if str(e) == "could not convert string to float: '-'":
                    print("Day-ahead prices not available")
                else:
                    print(e)

                return price_profile, None
        else:
            return price_profile, None
        
class PV:
    def __init__(self, col_name = metadata["pv col name"]):
        self.col_name = col_name

    def create_empty_pv_df(self, start_datetime):
        date_range_start = pd.date_range(start_datetime.strftime("%Y/%m/%d"), periods=24, freq='h')
        date_range_end = pd.date_range(start=date_range_start[1], periods=24, freq='h')
        #df = pd.DataFrame(np.full((24,3),0.0),columns=['Start date','End date', metadata["pv col name"]])
        df = pd.DataFrame(np.full((24,),0.0),columns=[metadata["pv col name"]],index=date_range_end)
        #df['Start date'] = date_range_start
        #df['End date'] = date_range_end
        return df

    def create_empty_caps_df(self, start_datetime):
        date_range_start = pd.date_range(start_datetime.strftime("%Y/%m/%d"), periods=24, freq='h')
        date_range_end = pd.date_range(start=date_range_start[1], periods=24, freq='h')
        df = pd.DataFrame(np.full((24,3),0.0),columns=['Start date','End date', metadata["pv cap col name"]])
        df['Start date'] = date_range_start
        df['End date'] = date_range_end
        return df

    def get_historic_capacity(self, year=2019):
        """
        Get the historic total PV system installed capacity in Germany.

        Remarks:
        The date range is 2018-2023
        The total installed capacity of Germany is 73.421 MW as per 2024.
        According to www.solarpowereurope.org the industrial scale size of pv system
        is between 250 to 1000 kW of capacity.
        """
        x = year-2019

        cap_value = 338.66*x + 6476.9

        return cap_value

    def get_historic_day_pv(self,year=2019, month=1, day=1):
        """
        Get a list with the total pv energy data [MWh] at 1-hour resolution for a given day

        Remarks: The total installed capacity of Germany is 73.421 MW as per 2024.
        According to www.solarpowereurope.org the industrial scale size of pv system
        is between 250 to 1000 kW of capacity.
        """
        pvs = []
        date = datetime(year, month, day)
        os_type = os.name
        df = pv_data
        for i in range(24):
            ts = date + timedelta(hours=i)

            # Change the format based on operative system
            if os_type == 'posix':
                date_str = ts.strftime("%b %-d, %Y %-I:%M %p")
            else:
                date_str = ts.strftime("%b %#d, %Y %#I:%M %p")

            row = df[df["Start date"] == date_str]
            row_value = row[self.col_name].values[0]

            # Remove commas from the string
            row_value = row_value.replace(',', '')

            try:
                pv_value = float(row_value)
            except Exception as e:
                print(e)
                pv_value = 0.0

            pvs.append(pv_value)

        return np.array(pvs)

    def get_pv_historic_potential(self,installed_capacity_in_MW, year=2019, month=1, day=1):
        cap = self.get_historic_capacity(year)
        pvs = self.get_historic_day_pv(year, month, day)
        scaled_pv_in_kW = pvs * installed_capacity_in_MW * 1000 / cap
        return scaled_pv_in_kW

    def get_pv_power(self,installed_capacity_in_MW, date, resolution="15min"):
        """
        Return a 1-minute or 15-minutes resolution timeseries with the power generated by a PV array with a given installed capacity.
        """
        year = date.year
        cap = self.get_historic_capacity(year)
        pv_df = self.get_pvs(date - timedelta(days=1))
        if pv_df is None:
            print("&&&& Could not get PV energy dataset for the current day")
            return None

        pvs = pv_df[metadata["pv col name"]].values
        scaled_pv_in_kW = pvs * installed_capacity_in_MW * 1000 / cap
        pv_df[metadata["pv col name"]] = scaled_pv_in_kW
        pv_df.columns = ["PV power [kW]"]

        # Resample to resolution
        if resolution == "1min" or resolution == "min":
            df_resampled = pv_df.resample('min').mean()
        else:
            df_resampled = pv_df.resample('15min').mean()

        # Forward fill to propagate last valid observation forward
        df_resampled['PV power [kW]'] = df_resampled['PV power [kW]'].ffill()

        #Drop the last row
        df_resampled = df_resampled.drop(df_resampled.index[-1])

        return df_resampled.values.squeeze()

    def get_pvs(self, date):
        """"
        Request energy marked data through the www.smard.de API.
        """

        # Define the URL to send the request to
        url = 'https://www.smard.de/nip-download-manager/nip/download/market-data'  # Replace with the actual URL

        # Define the headers for the request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*'
        }

        # Get the current day datetime starting at 00:00 hours
        picked_day = datetime(date.year, date.month, date.day)

        # Get the next day datetime starting at 00:00 hours (2 hours shift corrected)
        start_datetime = picked_day + timedelta(days=1)  # - timedelta(hours=2)
        print("start_datetime", start_datetime)

        # Get the other next day datetime starting at 00:00 hours
        end_datetime = start_datetime + timedelta(days=1)

        # Converting datetime to integer format
        start_date = int(start_datetime.timestamp() * 1000)
        end_date = int(end_datetime.timestamp() * 1000)

        # Define the data to be sent in the request
        data = {"request_form": [{"format": "CSV",
                                  "moduleIds": [2000122, 2005097, 2000715, 2003791, 2000123, 2000125],
                                  "region": "DE","timestamp_from": start_date, "timestamp_to": end_date,
                                  "type": "discrete", "language": "en", "resolution": ""}]}

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        # Send the POST request with the JSON data
        response = requests.post(url, headers=headers, data=json_data)

        # Check the response status code
        if response.status_code == 200:
            # Successful request

            # Decode the response content using utf-8-sig
            csv_str = response.content.decode("utf-8")
            list = [x.split(';') for x in csv_str.split('\n')]
            list[0][0] = 'Start date'
            # Convert string to data frame
            df = pd.DataFrame(list[1:-1])

            # Check if dataset is empty
            if df.iloc[0, 0] == 'No data for submitted query\r':
                print("&&&& No data for submitted query")
                return None #self.create_empty_pv_df(start_datetime)

            # Adding the headings
            df.columns = list[0][:df.shape[1]]

            # Converting strings into floats
            try:
                df[metadata["pv col name"]] = df[metadata["pv col name"]].str.replace(',', '').astype(float)
            except Exception as e:
                print("&&&& PV energy values unavailable due to: ",e)
                return None

            # Downsample to 1h resolution
            df['End date'] = pd.to_datetime(df['End date'])
            sub_df = df[["End date",metadata["pv col name"]]]
            sub_df = sub_df.resample('h', on='End date').sum()

            # Renaming the index column
            sub_df.index.name = "timestamp"

            return sub_df
        else:
            # Unsuccessful request
            print(f"Request failed with status code: {response.status_code}")
            print("Response content:", response.text)
            return self.create_empty_pv_df(start_datetime)

    def get_caps(self, date):
        """"
        Request energy marked data through the www.smard.de API.
        """

        # Define the URL to send the request to
        url = 'https://www.smard.de/nip-download-manager/nip/download/market-data'  # Replace with the actual URL

        # Define the headers for the request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*'
        }

        # Get the current day datetime starting at 00:00 hours
        picked_day = datetime(date.year, 1, 1)

        # Get the next day datetime starting at 00:00 hours (2 hours shift corrected)
        start_datetime = picked_day - timedelta(days=1)  # - timedelta(hours=2)
        print("start_datetime", start_datetime)

        # Get the other next day datetime starting at 00:00 hours
        end_datetime = start_datetime + timedelta(days=2)

        # Converting datetime to integer format
        start_date = int(start_datetime.timestamp() * 1000)
        end_date = int(end_datetime.timestamp() * 1000)

        # Define the data to be sent in the request
        data = {"request_form": [{"format": "CSV",
                                  "moduleIds": [3004073, 3004076, 3004072, 3004074, 3004075, 3000186, 3000188, 3000189,
                                                3000194,3000198, 3003792, 3000207], "region": "DE",
                                  "timestamp_from": start_date,
                                  "timestamp_to": end_date, "type": "discrete", "language": "en",
                                  "resolution": ""}]}

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        # Send the POST request with the JSON data
        response = requests.post(url, headers=headers, data=json_data)

        # Check the response status code
        if response.status_code == 200:
            # Successful request

            # Decode the response content using utf-8-sig
            csv_str = response.content.decode("utf-8")
            list = [x.split(';') for x in csv_str.split('\n')]
            list[0][0] = 'Start date'
            # Convert string to data frame
            df = pd.DataFrame(list[1:-1])

            # Check if dataset is empty
            if df.iloc[0, 0] == 'No data for submitted query\r':
                return self.create_empty_caps_df(start_datetime)

            # Adding the headings
            df.columns = list[0][:df.shape[1]]

            return df
        else:
            # Unsuccessful request
            print(f"Request failed with status code: {response.status_code}")
            print("Response content:", response.text)
            return self.create_empty_caps_df(start_datetime)
        
    def get_power_difference(self, pv_power, scheduler, date):

        # Get required power
        power = scheduler.get_power()
        required_power = power['Total power'].values.squeeze()

        savings = []
        for i in range(1440):
            if pv_power[i] > required_power[i]:
                savings.append(required_power[i])
            else:
                savings.append(pv_power[i])
                
        savings = np.array(savings)
        data = {
            "required power" : required_power.tolist(),
            "pv power" : pv_power.tolist(),
            "power from grid" : (required_power-savings).tolist(),
            "power offset" : savings.tolist()
        }

        print("pv_power", pv_power)

        index = pd.date_range(date,periods=1440,freq='min')

        return pd.DataFrame(data,index=index)
    
    def _get_power_difference(self, pv_power, schedule, date):

        # Get required power
        power = Scheduler()._get_power(schedule, date)
        required_power = power['Total power'].values.squeeze()


        savings = []
        for i in range(96):
            if pv_power[i] > required_power[i]:
                savings.append(required_power[i])
            else:
                savings.append(pv_power[i])
                
        savings = np.array(savings)
        data = {
            "required energy" : required_power.tolist(),
            "pv energy" : pv_power.tolist(),
            "energy from grid" : (required_power-savings).tolist(),
            "energy saved" : savings.tolist()
        }

        #print("pv_power", pv_power)

        index = pd.date_range(date,periods=96,freq='15min')

        return pd.DataFrame(data,index=index)
    
    def metrics(self, prices, schedule_list, savings_df, pw_df):
        _total_cost, _total_energy, _total_production, _total_number_of_jobs = EnergyPrices()._metrics(schedule_list, prices, pw_df)
        total_energy_saved, total_cost, pv_energy_generated = 0, 0, 0
        totals = savings_df.sum()
        pv_energy_generated = totals["pv energy"]/4
        total_energy_saved = totals["energy saved"]/4

        for i, power in enumerate(savings_df["energy from grid"]):
            price = prices[int(i/60)]/1000
            total_cost += power*price*(15/60)

        total_savings = _total_cost - total_cost
        delta_cost = int(-total_savings*100/_total_cost)
        total_energy_usage = _total_energy - total_energy_saved
        delta_energy = int(-total_energy_saved*100/_total_energy)

        return total_cost, delta_cost, total_energy_usage, delta_energy, total_savings, pv_energy_generated


class Helper:
    """
    Class for miscellaneous methods
    """
    def time_to_int(self, time, _timestep_size_seconds=60) -> int:
        """Converts `time` from pandas.Timestamp / pandas.Timedelta into the internal integer format - uses the globally set time unit (e.g. minutes / seconds)
    
        Args:
            time (_type_): Timestamp / Timedelta to convert
    
        Returns:
            int: Converted time
        """
        assert not (isinstance(time, int) or isinstance(time, np.integer)), "time_to_int: is already integer"
    
        if isinstance(time, pd.Timestamp):
            total_seconds = time.to_pydatetime().timestamp() # seconds since start of Unix epoch
        elif isinstance(time, pd.Timedelta):
            total_seconds = time.total_seconds()
        elif isinstance(time, datetime):
            total_seconds = time.timestamp()
        else:
            raise ValueError(f"time_to_int: unknown input format, type: {type(time)}")
        return int(total_seconds // _timestep_size_seconds) # somehow this results in a float when omitting int(...)? 


    def int_to_time(self, time: int, _timestep_size_seconds=60) -> pd.Timestamp:
        """Converts `time` from the internal integer format to pandas.Timestamp - uses the globally set time unit (e.g. minutes / seconds)
    
        Args:
            time (int): Timestamp to convert
    
        Returns:
            pd.Timestamp: Converted time
        """
        if isinstance(time, int) or isinstance(time, np.integer):
            return pd.Timestamp.fromtimestamp(time * _timestep_size_seconds)
        else:
            raise ValueError(f"time_to_int: already in time format / unknown input format? type: {type(time)}")
        
    def create_scheduler(filename="saved_sch",n_jobs = 30, n_machines = 3):
        scheduler = Scheduler().main(n_jobs, n_machines, "create")
        scheduler.create_schedule()
        scheduler.save()
        return scheduler

    def load_scheduler(filename="saved_sch"):
        scheduler = Scheduler().load()
        return scheduler

    def get_scheduler(self,load):
        if load:
            try:
                scheduler = self.load_scheduler()
            except Exception as e:
                print(e)
                scheduler = self.create_scheduler()
        else:
            scheduler = self.create_scheduler()

        return scheduler

    def timestamp_to_dayminutes(self,timestamp,duration):
        start = timestamp.hour*60+timestamp.minute
        end = start + duration
        return [start,end]

    def get_metadata(self):
        return metadata
