import torch
from .model import RosaModel
from .layer import RosaLayer
from .hooks import SaveInputHook, ManualGradCollectorHook
from typing import List, Dict
from transformers import TrainerCallback


try:
    from composer.core import Algorithm, Event
    COMPOSER_ALG_CLASS = Algorithm
    COMPOSER_EVENT_CLASS = Event
except ImportError:
    COMPOSER_ALG_CLASS = object
    COMPOSER_EVENT_CLASS = None


class RosaScheduler(TrainerCallback, COMPOSER_ALG_CLASS):
    def __init__(self, model: RosaModel) -> None:
        COMPOSER_ALG_CLASS.__init__(self)
        TrainerCallback.__init__(self)

        self._model = model

        config = model.peft_config
        assert len(config) == 1 and 'default' in config, 'only one default adapter is supported for now'
        config = config['default']

        self._mask_load_path = getattr(config, 'mask_load_path', None)
        self._mask_save_path = getattr(config, 'mask_save_path', None)
        self._spa_num_grads = getattr(config, 'spa_num_grads', 1)
        self._grad_acc_mode = getattr(config, 'grad_acc_mode', 'mean_squared')
        self._terminate_after_mask_generation = getattr(config, 'terminate_after_mask_generation', False)
        
        self._d = getattr(config, 'd', 0.)
        self._r = getattr(config, 'r', 0)

        assert None in [self._mask_load_path, self._mask_save_path], 'at least one of mask_save_path and mask_load_path has to be none.'
        if self._d > 0:
            if self._terminate_after_mask_generation:
                assert self._mask_save_path is not None
                assert self._mask_load_path is None

            if self._mask_load_path is not None:
                self._set_spa_masks(torch.load(self._mask_load_path))

        schedule_name = getattr(config, 'schedule', None)
        self._schedule = self._create_schedule(schedule_name)

        self._step = 0
        self._handles = []
    
    def _create_schedule(self, schedule_name: str) -> List[dict]:
        assert schedule_name is not None, "RoSA schedule has to be specified"

        if schedule_name in ['default', 'df']:
            return self._create_schedule('wl0')
        
        elif schedule_name == 'spa_only':
            assert self._d > 0, 'spa_only schedule requires density > 0'
            return self._generate_spa_schedule(self._mask_load_path is None)
        
        elif schedule_name == 'lora_only':
            assert self._d == 0, 'lora_only schedule requires density = 0'
            return self._generate_lora_schedule()
        
        elif schedule_name.startswith('wl'): # wl64 or wl224
            assert schedule_name == 'wl0' or self._d > 0, 'wl schedule requires density > 0'
            lora_warmup_steps = int(schedule_name.split('wl')[-1])
            return self._generate_wl_schedule(lora_warmup_steps, self._mask_load_path is None)
        else:
            assert False, f"RoSA schedule {schedule_name} is not implemented (df and ws schedules will be implemented later)."

    def _generate_spa_schedule(self, grad_colletion_needed: bool) -> List[dict]:
        schedule = []
        if grad_colletion_needed:
            schedule.append({'agenda': ['grad_collection'], 'end': self._spa_num_grads})
        schedule.append({'agenda': ['spa'], 'end': None})
        return schedule

    def _generate_lora_schedule(self) -> List[dict]:
        schedule = [{'agenda': ['lora'], 'end': None}]
        return schedule
    
    def _generate_wl_schedule(self, warmup: int, grad_colletion_needed: bool) -> List[dict]:
        schedule = []
        if warmup > 0:
            schedule.append({'agenda': ['lora'], 'end': warmup})
        if grad_colletion_needed:
            schedule.append({'agenda': ['lora', 'grad_collection'], 'end': warmup + self._spa_num_grads})
        schedule.append({'agenda': ['lora', 'spa'], 'end': None})
        return schedule

    def _get_agenda(self, step: int) -> List:
        for item in self._schedule:
            if item['end'] is None or step < item['end']:
                return item['agenda']
        assert False, f"no agenda for step {step}"

    def _get_current_agenda(self) -> List:
        return self._get_agenda(self._step)

    def _get_prev_agenda(self) -> List:
        return self._get_agenda(self._step - 1) if self._step > 0 else None

    def _get_next_agenda(self) -> List:
        return self._get_agenda(self._step + 1)
    
    def _set_spa_masks(self, masks: Dict[str, torch.Tensor]) -> None:
        self._model.set_spa_masks(masks)
    
    # methods for the composer Algorithm interface
    def match(self, event, state):
        if COMPOSER_EVENT_CLASS is None:
            return False
        return event in [COMPOSER_EVENT_CLASS.BEFORE_TRAIN_BATCH, COMPOSER_EVENT_CLASS.AFTER_TRAIN_BATCH]

    def apply(self, event, state, logger):
        if COMPOSER_EVENT_CLASS is None:
            return
        
        if event == COMPOSER_EVENT_CLASS.BEFORE_TRAIN_BATCH:
            self._on_step_begin()
        elif event == COMPOSER_EVENT_CLASS.AFTER_TRAIN_BATCH:
            self._on_step_end()

    # methods for the transformers TrainerCallback interface
    def on_step_begin(self, args, state, control, **kwargs):
        self._on_step_begin()

    def on_step_end(self, args, state, control, **kwargs):
        self._on_step_end()

    # main methods
    @torch.no_grad()
    def _on_step_begin(self):
        agenda = self._get_current_agenda()
        print('AGENDA', agenda)

        for _, p in self._model.named_parameters():
            p.requires_grad = False

        if self._mask_load_path is not None and not self._model.spa_activated:
            print('loading masks')
            masks = torch.load(self._mask_load_path)
            self._set_spa_masks(masks) # this activates spa

        for name, module in self._model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            weight = module.find_weight()
            if 'grad_collection' in agenda and not self._model.spa_activated:
                # if weight.is_floating_point:
                #     weight.requires_grad = True
                #     handle = weight.register_post_accumulate_grad_hook(GradCollectorHook(name, module, self._grad_acc_mode))
                #     self._handles.append(handle)

                # the weight cannot require grad if it's not floating point
                # we employ two hooks to capture input and grad_output then
                # multiply the two to get the gradients
                handle1 = module.register_forward_hook(SaveInputHook(name, module))
                handle2 = module.register_full_backward_hook(ManualGradCollectorHook(name, module, self._grad_acc_mode))
                self._handles.append(handle1)
                self._handles.append(handle2)
            else:
                if weight.is_floating_point:
                    weight.requires_grad = False
            
            module.set_lora_requires_grad('lora' in agenda)
            
            if self._model.spa_activated:
                module.set_spa_requires_grad('spa' in agenda)

    @torch.no_grad()
    def _on_step_end(self):
        agenda = self._get_current_agenda()
        next_agenda = self._get_next_agenda()

        if not self._model.spa_activated and 'grad_collection' in agenda and 'grad_collection' not in next_agenda:
            print('finished collecting gradients')
            self._generate_masks_and_activate_spa(self._model)

        for handle in self._handles:
            handle.remove()
        
        self._handles = []
        self._step += 1
    
    @torch.no_grad()
    def _grad_to_mask_fn(self, grad):
        idx = torch.topk(torch.abs(grad.flatten()).float(), int(self._d * grad.numel()), sorted=False).indices
        mask = torch.zeros_like(grad.flatten())
        mask.scatter_(0, idx, 1.)
        mask = mask.reshape_as(grad).bool()
        return mask

    @torch.no_grad()
    def _generate_masks_and_activate_spa(self, model):
        print('generating masks and activating spa')
        assert self._d > 0, 'mask generation requires spa density to be > 0'

        masks = {}
        for name, module in model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            assert hasattr(module, 'collected_grad'), 'target module must have a collected_grad for mask generation, something is wrong!'
            print(f'generating spa mask for {name} with {module.collected_grad_cnt} grads.')

            masks[name] = self._grad_to_mask_fn(module.collected_grad)
            delattr(module, 'collected_grad')
            delattr(module, 'collected_grad_cnt')
            if hasattr(module, 'saved_input'):
                delattr(module, 'saved_input')
        
        if self._mask_save_path is not None:
            print('saving the masks...')
            torch.save(masks, self._mask_save_path)
            print('masks saved.')
        
        if self._terminate_after_mask_generation:
            print('Done. halting...')
            raise SystemExit()
        else:
            self._set_spa_masks(masks)


