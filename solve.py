import os
import pandas as pd
import numpy as np
from utils import read_csv_decorator, get_input_file_dict
from messages import MyInfo, MyWarning, MyError
from app_logger import get_logger
from configparser import NoOptionError

"""Обернем декоратором функцию pd.read"""
pd_read_csv = read_csv_decorator(pd.read_csv)

class Solve():
    def __init__(self):
        """путь к директории исходных данных"""
        self.input_dir_path = None
        """путь к директории решения"""
        self.solve_path = self._get_solve_dir()
        
        """по умолчанию содержит минимальный перечень анализируемых компонентов смеси"""
        self.list_components = ["H2","O2","VAPOUR","N2"]
        
        """список горючих компонентов"""
        self.list_posible_fuels = ('CO', 'H2')

        """среднеобъемная температура; pandas.DataFrame (columns = [Time, Tvol])"""
        self.t_vol_df = None
        """давление в реперном объеме (config.inin, Nbox_by_p_monitor); pandas.DataFrame (columns = [Time, Tvol])"""
        self.parameters_df = None
        
        """массы/мольные концентрации компонентов; pandas.DataFrame"""
        self.components_df = None
        
        """Создание логера """
        logger = get_logger("{}\\{}".format(self.solve_path, "log.log"))

        """Создание сообщений """
        self.myInfo = MyInfo(logger)
        self.myError = MyError(logger)
        self.myWarning = MyWarning(logger)

        self.myInfo.send('Program is started')

    def check_pd_series_length(self, df):
        """проверка того, что pd.DataFrame.series имеют одинаковую длину"""
        norma_series = df.count()
        if not np.all(norma_series == norma_series[0]):
            message_list = ['column {} has length {}'.format(key, value) \
                            for key, value in norma_series.items() if key != 'Time']
            message = 'Check input files. Dataframe using by program have columns with different length:\n' +\
                   '\n'.join(['\t\t\t\t--->\t'+item for item in message_list])
            return self.myError.send(message)

    @staticmethod        
    def _get_solve_dir():    
        """ созданиt нового каталога для каждого запуска скрипта"""
        workdir_path = os.getcwd()
        solve_path = None
        i = 0
        while 1:
            solve_path = '{}\\solve\\{}'.format(workdir_path, i)
            try:
                os.chdir(solve_path)
                """Переход в корневую папку"""
                os.chdir('..\\')
                os.chdir('..\\')
                i += 1
            except FileNotFoundError:
                os.makedirs(solve_path)
                return solve_path
            
    def get_input_dir_path(self, config):
        self.input_dir_path = config["general"]["INPUT_DIR_PATH"]
        
    def set_multiindex(self):
        mass_columns = [c for c in self.components_df.columns if 'Time' not in c]
        self.components_df = self.components_df.rename(columns={component: (component, 'begin') for component in mass_columns})
        self.components_df = self.components_df.rename(columns={"Time": ("Time", 'time')})
        self.components_df.columns = pd.MultiIndex.from_tuples(self.components_df.columns)   
        
    def set_tvol(self, t_file, volumes):
        """ возвращает среднеобъемную температуру"""
        """чтение файла температур """
        self.myInfo.send("---> {} is reading".format(t_file))
        
        path_to_t_file = '{}\\{}'.format(self.input_dir_path, t_file)
        t_vol_df = pd_read_csv(path_to_t_file)
        t_columns = [c for c in t_vol_df.columns if 'T_box_' in c]
        
        if len(t_columns) != len(volumes):
            self.myError.send("Number of boxes in BXDATA.DAT doesnt match with number of columns in {}".format(t_file))
        
        """ оставляем только нужные колонки, важно для последующей группировки"""
        t_vol_df = t_vol_df[t_columns + ['Time']]
        
        """относительные объемы"""
        volumes = [float(v) for v in volumes]
        rel_volumes = [v/sum(volumes) for v in volumes]
        
        """температура * весовой коэффициент"""
        t_vol_df[t_columns] = t_vol_df[t_columns]*rel_volumes
        """агрегация"""
        t_vol_df = t_vol_df.groupby(lambda c: (c in t_columns), axis=1).agg('sum')
        
        """ Корректировка имен столбцов. Запись данных в dataframe параметров среды """
        self.parameters_df = t_vol_df.rename(columns={False: 'Time', True: 'Tvol'})
        t_vol_df = pd.DataFrame()
        """ переводим среднеобъемную температуру в градусы Кельвина"""
        self.parameters_df['Tvol'] += 273.15
        
    def set_p(self, p_file, config):
        """ возвращает давление"""
        """чтение файла давления """
        self.myInfo.send("---> {} is reading".format(p_file))
        path_to_p_file = '{}\\{}'.format(self.input_dir_path, p_file)
        p_df = pd_read_csv(path_to_p_file)
        
        ref_box_for_p = config['additional']['Nbox_by_p_monitor']
        try:
            self.parameters_df["pressure"] = p_df['P_box__{}'.format(ref_box_for_p)]
        except KeyError:
            self.myError.send("Box number for pressure monitor in config.ini does not match with number or names of columns {}".format(p_file))
         
    def read_concentrations(self, input_dir_dict_comp):
        """чтение данных концентраций из файлов"""
        mass_dict_df = {}
        
        for key, file_name in input_dir_dict_comp.items():
            try:
                self.myInfo.send("---> {} is reading".format(file_name))
                path_to_M_file = '{}\\{}'.format(self.input_dir_path, file_name)
                mass_dict_df[key] = pd_read_csv(path_to_M_file)
                """ перечень колонок с массами из читаемого файла"""
                mass_columns = [c for c in mass_dict_df[key].columns if 'M_box_' in c]
                mass_dict_df[key] = mass_dict_df[key].groupby(lambda c: (c in mass_columns), axis=1).agg('sum')
                mass_dict_df[key] = mass_dict_df[key].rename(columns={False: 'Time', True: 'SummMass'})
    
            except FileNotFoundError:
                self.myError.send("xxxxX {} is not readed. Try again".format(file_name))
                
        return mass_dict_df
            
    def _create_components_df(self, mass_dict_df):
        self.components_df = pd.DataFrame({key.split('_')[0].upper(): value["SummMass"] for key,value in mass_dict_df.items()})
        self.components_df["Time"] = mass_dict_df[list(mass_dict_df.keys())[0]]["Time"] # добавление столбца времени

    def set_concentrations(self, input_dir_dict_comp, input_dir_list):
        mass_dict_df = self.read_concentrations(input_dir_dict_comp)
        input_dir_dict__CO = get_input_file_dict(input_dir_list,('CO_m',))
        """ CO необязательный компонент для анализа"""
        if input_dir_dict__CO != None:
            mass_dict_df.update(self.read_concentrations(input_dir_dict__CO))
            mass_dict_df['CO2_m'] = pd.DataFrame({'Time': mass_dict_df['CO_m']['Time'],
                                                'SummMass':[0 for i in mass_dict_df['CO_m']['Time'].index]})
            self.list_components += ['CO','CO2']
        self._create_components_df(mass_dict_df)
        
    def get_mol_fraction(self, config):
        molar_mass = {material.upper(): float(config.get('molar mass', material)) for material in config['molar mass']\
                      if material.upper() in self.list_components}
        try:
            self.components_df[self.list_components] = self.components_df[self.list_components]/[molar_mass[key] for key in self.list_components]
        except KeyError:
            self.myError.send("Molar mass shoud be defined in config.ini for the following components: {}".format(self.list_components))
        self.components_df = self.components_df[["Time"] + self.list_components] #порядок столбцов с "Time"
        
    @staticmethod 
    def get_mass_fraction(components_df, config):
        """нужно преобразовывать при выводе не меняя исходный pd.DataFrame"""
        molar_mass = {material.upper(): float(config.get('molar mass', material)) for material in config['molar mass']}
        """ удаление колонок с суммарным числом моль (из них нельзя получить суммарную массу)"""
        components_df = components_df.drop(columns = [c for c in components_df.columns if 'Summ' in c])
        """ кортежи индексов колонок в файле выдачи"""
        mass_columns = [c for c in components_df.columns if 'Time' not in c]
        components_df[mass_columns] = components_df[mass_columns]*[molar_mass[key[0]] for key in mass_columns]
        components_df[('Summ','end')] = components_df.filter(regex = 'end', axis=1).groupby(all, axis=1).agg('sum')
        components_df[('Summ','begin')] = components_df.filter(regex = 'begin', axis=1).groupby(all, axis=1).agg('sum')
        return components_df
        
    def _get_new_concentration_CO(self): #-> df_n
        """ Реакция: 2CO + O2 = 2CO2
        nu_O2 - число моль O2 после сгорания всего CO: """
        self.components_df[('O2', 'end')] = self.components_df[('O2', 'end')] - 2*self.components_df[('CO', 'begin')]
        
        """из моля CO получается моль CO2; число моль CO2 равняется:
        что получилось из CO (все CO прореагировало) за вычетом того
        CO, которое непрореагировало, см. дальше"""
        self.components_df[('CO2', 'end')] = self.components_df[('CO', 'begin')] + [min(0,a)/2 for a in self.components_df[('O2', 'end')]]
        
        """если nu_O2 отрицательная, то это чмсло моль непрореагиравшего CO;
        в остальных случаях концентраци CO будет 0"""
        self.components_df[('CO', 'end')] = [- min(0, a)/2 for a in self.components_df[('O2', 'end')]]

    def _get_new_concentration_H2(self):  # -> df_n
        """ Реакция : 2H2 + O2 = 2H2O
        nu_O2_ - число моль O2 после сгорания всего CO и всего водорода. Реакция с CO рассматривается
        в первую очередь, поскольку CO активнее H2. Логика аналогична логике по горению CO"""
        nu_O2__ = self.components_df[('O2', 'end')] - 2*self.components_df[('H2', 'begin')]
        self.components_df[('H2', 'end')] = [- min(0, a)/2 for a in nu_O2__]
        self.components_df[('O2', 'end')] = [max(0, a) for a in nu_O2__]

    def get_new_concentrations(self):
        self.components_df[('VAPOUR', 'end')] = self.components_df[('VAPOUR', 'begin')]
        self.components_df[('N2', 'end')] = self.components_df[('N2', 'begin')]
        self.components_df[('O2', 'end')] = self.components_df[('O2', 'begin')]
        """ CO необязательный компонент для анализа"""
        if ('CO', 'begin') in self.components_df.columns.tolist():
            self._get_new_concentration_CO()
        self._get_new_concentration_H2()

    @staticmethod
    def _get_CV(Coeff_Cp, adiabatic_power): #dict -> dict
        for key in Coeff_Cp.keys():
            Coeff_Cp[key] = list(map(lambda x: x/adiabatic_power[key], Coeff_Cp[key]))
        return Coeff_Cp
    
    @staticmethod  
    def _get_C_by_polinom(C_coefficients, T_vol):
        powers_list = (0,-2,1,2,3)#показатели степеней полинома
        return sum(map(lambda C,k: round(C*T_vol**k,6), C_coefficients, powers_list))
    
    @staticmethod
    def _get_T_AICC(Gener_poly_coeffs, Q):
        '''Имеем уравнение: (nu_1*Cp_1(T)+ ... +nu_N*Cp_N(T))*T*1000 = (H0+H_H2_burn+H_H2_burn) замена переменных Т/1е+04 -> T, меняем коэффициенты
        чтобы получить полином с целыми степенями, домнажаем обе части уравнения на Т т.е.(nu_1*Cp_1(T)+ ... +nu_N*Cp_N(T))умножается на T^2'''
        Gener_poly_coeffs = list(Gener_poly_coeffs.c)

        Gener_poly_coeffs_n = []
        Gener_poly_coeffs_n.append(Gener_poly_coeffs[4]/1e+12)
        Gener_poly_coeffs_n.append(Gener_poly_coeffs[3]/1e+08)
        Gener_poly_coeffs_n.append(Gener_poly_coeffs[2]/1e+04)
        Gener_poly_coeffs_n.append(Gener_poly_coeffs[0])
        Gener_poly_coeffs_n.append(-Q)
        Gener_poly_coeffs_n.append(Gener_poly_coeffs[1]*1e+08)
            
        Big_poly = np.poly1d(Gener_poly_coeffs_n)
        TRoot =  list([round(float(r),6) for r in Big_poly.r if 273.15 < r < 1500])[0]
        '''298.15 < r < 1500 коэффициенты Coeff_Cp работают для определения Сp
        в определенном диапазоне температур,  от 298.15 и в среднем до 1500 К''' 
        return TRoot

    def get_heat_of_combustion(self, config):
        for fuel_component in self.list_posible_fuels:
            try:
                self.parameters_df['H_'+fuel_component] = (self.components_df[(fuel_component,'begin')] - self.components_df[(fuel_component,'end')])\
                                                        *float(config['heat of cumbustion']['dH_'+fuel_component])*1000
                self.myInfo.send('---> heat of cumbustion for {} is determined'.format(fuel_component))
            except KeyError:
                self.myWarning.send('---> there is no {} data. Are you sure that is in-vessel stage calculation only?'.format(fuel_component))
            
    def get_P_AICC(self, config):
        """получение списка показателей адиабаты для компонентов смеси"""
        try:
            adiabatic_powers = {component.upper(): float(config.get('adiabatic exponent', component)) for component in self.list_components}
        except NoOptionError: 
            self.myError.send("Adiabatic powers for Cp determining shoud be defined in config.ini for the following components: {}".format(self.list_components))
        """получение списка списков коэффициентов полиномов для определения коэффициентов изобарной теплоемкости для каждого компонента газовой смеси"""
        try:
            C_coeff_for_polinoms ={component.upper(): config.getlist('Coeff_Cp', component) for component in self.list_components}
        except NoOptionError:
            self.myError.send("Polinom coefficients for Cp determining shoud be defined in config.ini for the following components: {}".format(self.list_components))
            
        """получение списка списков коэффициентов полиномов для определения коэффициентов изохорной теплоемкости для каждого компонента газовой смеси"""
        C_coeff_for_polinoms = self._get_CV(C_coeff_for_polinoms, adiabatic_powers)

        """ Создание массивов для определения теплоемкости: массив коэффициентов для определения теплоемкости умножается на количество вещества
        на каждом временном шаге для каждого компонента газовой смеси """
        """ базовая агрегирующая лямбда-функция для определения коэффициентов, умноженных на количества моль"""
        lambda_poly_coeffs_1 = lambda series, inp_arr: [c*it*1000 for c in inp_arr for it in series]
        
        """агрегирующая функция"""
        def nppoly_for_items(series, inp_arr):
            """возвращает объект np.poly1d от массива коэффициентов для определения теплоемкости умноженных на количества моль"""
            return np.poly1d(lambda_poly_coeffs_1(series, inp_arr))
        
        #передавать аргумент (в моем случае ключ для словаря с соответствующими компоненту коэффициентами) можно,
        #если обернуть lambda-функцию, принимающую pd.Series, в lambda-функцию, принимающую ключ словаря
        """ лямбда-фукции дающие объект np.poly1d или массив коэффициентов соответственно для каждого набора коэффициентов;
        значения коэффициентов в наборах зависят от компонета газовой смеси"""
        lambda_poly_coeffs_nppoly = lambda component: lambda series: nppoly_for_items(series, C_coeff_for_polinoms[component])
        lambda_poly_coeffs_list = lambda component: lambda series: lambda_poly_coeffs_1(series, C_coeff_for_polinoms[component])

        """создание словаря "название колонки": агрегирующая функция"""
        agg_func_selection = {(component, 'end'): ['mean', lambda_poly_coeffs_nppoly(component)] for component in self.list_components}
        agg_func_selection[('Time', 'time')] = ['mean']
        agg_func_selection.update({(component, 'begin'): ['mean', lambda_poly_coeffs_list(component)] for component in self.list_components})
        """агрегирование"""
        self.components_df = self.components_df.groupby([('Time', 'time')]).agg(agg_func_selection)
        
        """ определение энтальпий (до учета AICC) для каждого компонента газовой смеси"""
        for component in self.list_components:
            self.parameters_df['H0_{}'.format(component)] = list(map(lambda C_list, Tvol: self._get_C_by_polinom(C_list, Tvol/10000)*Tvol,\
                                                                        self.components_df[(component, 'begin', '<lambda_0>')], self.parameters_df['Tvol']))
        """ удаление колонок набора коэффициентов для определения теплоемкости каждого компонента газовой смеси """
        self.components_df = self.components_df.drop(columns = [(component, 'begin', '<lambda_0>') for component in self.list_components])
        """ определение энтальпии смеси до учета AICC """
        self.parameters_df['H0'] = self.parameters_df.filter(regex = 'H0', axis=1).groupby(all, axis=1).agg('sum')
        """ удаление колонок энтальпии каждого компонента газовой смеси """
        self.parameters_df = self.parameters_df.drop(columns = ['H0_{}'.format(component) for component in self.list_components])
        self.myInfo.send('---> mixture enthalphy is determined')
        
        """ Определение T_AICC """
        """суммировние полиномов, создание полинома коэффициентов для определения (теплоемкости) температуры после учета AICC"""
        self.components_df[('Summ','end','<lambda_0>')] =  self.components_df.filter(regex = 'lambda', axis=1).groupby(all, axis=1).agg('sum')
        """удаление колонок коэффициентов полинома (после учета AICC) для каждого компонента газовой смеси"""
        self.components_df = self.components_df.drop(columns = [(component, 'end', '<lambda_0>') for component in self.list_components])
        """ Определение температуры смеси c учетом AICC """
        Q_burn = sum([self.parameters_df['H_'+fuel] for fuel in list(set(self.list_components)&set(self.list_posible_fuels))])
        self.parameters_df['T_AICC'] = list(map(lambda C,Q: self._get_T_AICC(C,Q), self.components_df[('Summ','end','<lambda_0>')], self.parameters_df['H0']+Q_burn))
        self.myInfo.send('---> T_AICC is determined')
        """удаление колонки суммированных коэффициентов полиномов"""
        self.components_df = self.components_df.drop(columns = [('Summ','end','<lambda_0>')])
        """удаление ставшего лишним уровня в названиях колонок"""
        self.components_df.columns = self.components_df.columns.droplevel(2)
        self.components_df = self.components_df.reset_index(drop=True)
        """определения суммарного числа молей смеси"""
        self.components_df[('Summ', 'begin')] = self.components_df.filter(regex = 'begin', axis=1).groupby(all, axis=1).agg('sum')
        self.components_df[('Summ', 'end')] = self.components_df.filter(regex = 'end', axis=1).groupby(all, axis=1).agg('sum')
        """ Определение давления с учетом AICC"""
        self.parameters_df['P_AICC'] = self.parameters_df['pressure']*self.parameters_df['T_AICC']/self.parameters_df['Tvol']*self.components_df[('Summ', 'end')]/self.components_df[('Summ', 'begin')]
        self.myInfo.send('---> P_AICC is determined')
             

            